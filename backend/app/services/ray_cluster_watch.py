"""Log CDSW worker launch and Ray cluster node join events to job framework logs."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.services.job_log import append_job_log
from app.services.log_flair import (
    framework_line,
    ray_cluster_ready_line,
    ray_node_joined_line,
    worker_launch_line,
)

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def framework_log(db: Session, job_id: int, message: str) -> None:
    append_job_log(db, job_id, "framework", message)


def log_cdsw_worker_ids(db: Session, job_id: int, worker_ids: list[str], head: str) -> None:
    framework_log(db, job_id, framework_line("master", f"Ray master (head) at {head}"))
    total = len(worker_ids)
    for idx, wid in enumerate(worker_ids, start=1):
        framework_log(db, job_id, worker_launch_line(idx, total, wid, head))


def log_cdsw_worker_api_status(
    db: Session,
    job_id: int,
    list_workers_fn: Callable[[], list[dict[str, Any]]] | None,
    worker_ids: list[str],
) -> None:
    if not list_workers_fn or not worker_ids:
        return
    try:
        rows = list_workers_fn()
    except Exception as exc:  # noqa: BLE001
        framework_log(db, job_id, framework_line("warn", f"CDSW worker status API: {exc}"))
        return
    by_id = {str(r.get("id", r.get("worker_id", ""))): r for r in rows if isinstance(r, dict)}
    for wid in worker_ids:
        row = by_id.get(str(wid))
        if not row:
            framework_log(
                db,
                job_id,
                framework_line("info", f"worker {wid}: provisioning (not in API list yet)"),
            )
            continue
        status = row.get("status") or row.get("state") or "unknown"
        framework_log(
            db,
            job_id,
            framework_line("worker", f"worker {wid} platform status={status}"),
        )


def _alive_ray_nodes() -> list[dict[str, Any]]:
    import ray

    if not ray.is_initialized():
        return []
    out: list[dict[str, Any]] = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        out.append(node)
    return out


def _node_label(node: dict[str, Any]) -> str:
    addr = node.get("NodeManagerAddress") or node.get("node_ip_address") or "?"
    res = node.get("Resources") or {}
    gpu = res.get("GPU", 0)
    cpu = res.get("CPU", 0)
    return f"{addr} CPUs={cpu} GPUs={gpu}"


def log_ray_nodes_snapshot(log: LogFn, label: str) -> int:
    nodes = _alive_ray_nodes()
    log(framework_line("ray", f"{label}: {len(nodes)} alive node(s)"))
    for i, node in enumerate(nodes, start=1):
        log(framework_line("info", f"  node {i}: {_node_label(node)}"))
    return len(nodes)


def ray_cluster_snapshot_subprocess(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 45.0,
) -> dict[str, float] | None:
    """
    Cluster node/GPU snapshot without ``ray.init`` in the API / job-runner process.
    """
    from app.services.ray_driver_connect import (
        co_located_subprocess_ray_init_kwargs,
        ensure_ray_auth_token,
    )

    token = (auth_token or ensure_ray_auth_token(gcs_address) or "").strip()
    init_kw = co_located_subprocess_ray_init_kwargs(gcs_address)
    script = f"""
import json
import os
import ray

os.environ["RAY_ADDRESS"] = {json.dumps(gcs_address)}
if {json.dumps(token)}:
    os.environ["RAY_AUTH_TOKEN"] = {json.dumps(token)}
ray.init(**{json.dumps(init_kw)})
alive = 0
gpu = 0.0
cpu = 0.0
for n in ray.nodes():
    if not n.get("Alive"):
        continue
    alive += 1
    res = n.get("Resources") or {{}}
    gpu += float(res.get("GPU", 0) or 0)
    cpu += float(res.get("CPU", 0) or 0)
print(json.dumps({{"alive": alive, "gpu": gpu, "cpu": cpu}}))
ray.shutdown()
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        data = json.loads(lines[-1])
        if not isinstance(data, dict):
            return None
        return {
            "alive": float(data.get("alive", 0)),
            "gpu": float(data.get("gpu", 0)),
            "cpu": float(data.get("cpu", 0)),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def count_alive_ray_nodes_subprocess(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 45.0,
) -> int | None:
    snap = ray_cluster_snapshot_subprocess(
        gcs_address, auth_token=auth_token, timeout_sec=timeout_sec
    )
    if snap is None:
        return None
    return int(snap["alive"])


def wait_for_gcs_rpc_subprocess(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 120.0,
    poll_sec: float = 3.0,
) -> bool:
    """True when a subprocess can complete ``ray.init`` + list nodes (not just TCP)."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if ray_cluster_snapshot_subprocess(
            gcs_address, auth_token=auth_token, timeout_sec=45.0
        ) is not None:
            return True
        time.sleep(poll_sec)
    return False


def wait_for_cluster_nodes_subprocess(
    gcs_address: str,
    log: LogFn,
    *,
    min_alive: int,
    min_gpu: float = 0.0,
    timeout_sec: float = 600.0,
    poll_sec: float = 8.0,
    auth_token: str | None = None,
) -> int:
    """Poll Ray GCS until ``min_alive`` nodes (and optional GPUs) are available."""
    seen_counts: set[int] = set()
    deadline = time.monotonic() + timeout_sec
    last_heartbeat = time.monotonic()
    need_gpu = min_gpu > 0
    log(
        framework_line(
            "ray",
            f"Waiting for Ray cluster (need >={min_alive} node(s)"
            + (f", >={min_gpu:g} GPU" if need_gpu else "")
            + f", timeout {int(timeout_sec)}s)…",
        )
    )
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_heartbeat >= 30.0:
            last_heartbeat = now
            log(
                framework_line(
                    "ray",
                    "Still waiting for Ray cluster (GCS RPC + worker join)…",
                )
            )
        snap = ray_cluster_snapshot_subprocess(
            gcs_address, auth_token=auth_token, timeout_sec=40.0
        )
        if snap is not None:
            count = int(snap["alive"])
            if count not in seen_counts:
                seen_counts.add(count)
                log(
                    framework_line(
                        "ray",
                        f"Cluster reports {count} alive node(s), "
                        f"GPU={snap['gpu']:g} CPU={snap['cpu']:g}",
                    )
                )
            nodes_ok = count >= min_alive
            gpu_ok = not need_gpu or snap["gpu"] >= min_gpu
            if nodes_ok and gpu_ok:
                log(ray_cluster_ready_line(count))
                return count
        time.sleep(poll_sec)
    final = ray_cluster_snapshot_subprocess(
        gcs_address, auth_token=auth_token, timeout_sec=40.0
    )
    final_n = int(final["alive"]) if final else 0
    final_gpu = final["gpu"] if final else 0.0
    log(
        framework_line(
            "warn",
            f"Cluster wait timeout: {final_n} node(s), {final_gpu:g} GPU "
            f"(wanted >={min_alive} nodes"
            + (f", >={min_gpu:g} GPU" if need_gpu else "")
            + ")",
        )
    )
    return final_n


def wait_for_ray_nodes(
    log: LogFn,
    *,
    min_alive: int,
    timeout_sec: float = 600.0,
    poll_sec: float = 5.0,
) -> int:
    seen_addrs: set[str] = set()
    deadline = time.monotonic() + timeout_sec
    log(
        framework_line(
            "ray",
            f"Polling master for workers (need >={min_alive} nodes, timeout {int(timeout_sec)}s)",
        )
    )
    while time.monotonic() < deadline:
        nodes = _alive_ray_nodes()
        for node in nodes:
            addr = str(node.get("NodeManagerAddress") or "")
            if addr and addr not in seen_addrs:
                seen_addrs.add(addr)
                log(ray_node_joined_line(_node_label(node)))
        if len(nodes) >= min_alive:
            log(ray_cluster_ready_line(len(nodes)))
            return len(nodes)
        time.sleep(poll_sec)
    final = len(_alive_ray_nodes())
    log(
        framework_line(
            "warn",
            f"Cluster wait timeout: {final} node(s) alive (wanted >={min_alive})",
        )
    )
    return final
