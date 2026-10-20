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


def count_alive_ray_nodes_subprocess(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 45.0,
) -> int | None:
    """
    Count alive Ray nodes without ``ray.init`` in the API / job-runner process.

    Spawns a short-lived child Python process so the FastAPI pod never becomes a
    Ray driver client (avoids co-located head/driver bugs on CAI).
    """
    from app.services.ray_driver_connect import ensure_ray_auth_token

    token = (auth_token or ensure_ray_auth_token() or "").strip()
    script = f"""
import os
import ray

os.environ["RAY_ADDRESS"] = {json.dumps(gcs_address)}
if {json.dumps(token)}:
    os.environ["RAY_AUTH_TOKEN"] = {json.dumps(token)}
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
alive = sum(1 for n in ray.nodes() if n.get("Alive"))
print(alive)
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
        return int(lines[-1])
    except ValueError:
        return None


def wait_for_cluster_nodes_subprocess(
    gcs_address: str,
    log: LogFn,
    *,
    min_alive: int,
    timeout_sec: float = 600.0,
    poll_sec: float = 8.0,
    auth_token: str | None = None,
) -> int:
    """Poll Ray GCS until ``min_alive`` nodes are up (head + CDSW workers)."""
    seen_counts: set[int] = set()
    deadline = time.monotonic() + timeout_sec
    log(
        framework_line(
            "ray",
            f"Waiting for Ray cluster nodes (need >={min_alive}, timeout {int(timeout_sec)}s)…",
        )
    )
    while time.monotonic() < deadline:
        count = count_alive_ray_nodes_subprocess(
            gcs_address, auth_token=auth_token, timeout_sec=40.0
        )
        if count is not None and count not in seen_counts:
            seen_counts.add(count)
            log(framework_line("ray", f"Cluster reports {count} alive node(s)"))
        if count is not None and count >= min_alive:
            log(ray_cluster_ready_line(count))
            return count
        time.sleep(poll_sec)
    final = count_alive_ray_nodes_subprocess(
        gcs_address, auth_token=auth_token, timeout_sec=40.0
    )
    final_n = final if final is not None else 0
    log(
        framework_line(
            "warn",
            f"Cluster wait timeout: {final_n} node(s) alive (wanted >={min_alive})",
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
