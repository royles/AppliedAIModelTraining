"""Log CDSW worker launch and Ray cluster node join events to job framework logs."""

from __future__ import annotations

import json
import logging
import os
import re
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

_CLUSTER_SNAPSHOT_TIMEOUT_SEC = 18.0
_CLUSTER_WAIT_HEARTBEAT_SEC = 15.0
_CLUSTER_WAIT_POLL_SEC = 5.0


def _ray_status_subprocess(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 20.0,
) -> tuple[int, str, str]:
    """Run ``ray status``; return (exit_code, stdout, stderr)."""
    from app.services.ray_driver_connect import _ray_cli

    env = os.environ.copy()
    env["RAY_ADDRESS"] = gcs_address
    if auth_token:
        env["RAY_AUTH_TOKEN"] = auth_token
    else:
        env.pop("RAY_AUTH_TOKEN", None)
    try:
        proc = subprocess.run(
            _ray_cli("status"),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return -1, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def parse_ray_status_active_nodes(stdout: str) -> int | None:
    """Count node entries under the ``Active:`` section of ``ray status`` output."""
    lines = stdout.splitlines()
    in_active = False
    count = 0
    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if "active:" in lower and "pending" not in lower:
            in_active = True
            continue
        if in_active:
            if lower.startswith("pending:") or lower.startswith("recent failures:"):
                break
            if not line:
                continue
            if "node_" in line or line[0].isdigit():
                count += 1
    return count if in_active else None


def parse_ray_status_resource_capacities(stdout: str) -> dict[str, float]:
    """Parse ``Total Usage`` capacity lines (e.g. ``0.0/1.0 GPU`` → 1 GPU)."""
    caps = {"cpu": 0.0, "gpu": 0.0}
    in_totals = False
    for raw in stdout.splitlines():
        line = raw.strip()
        lower = line.lower()
        if "total usage:" in lower:
            in_totals = True
            continue
        if in_totals:
            if lower.startswith("from request_resources:") or lower.startswith("pending demands:"):
                break
            match = re.match(r"[\d.]+/([\d.]+)\s+(CPU|GPU)\b", line, re.IGNORECASE)
            if match:
                caps[match.group(2).lower()] = float(match.group(1))
    return caps


def ray_status_cluster_snapshot(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 20.0,
) -> dict[str, float] | None:
    """Node/GPU counts from ``ray status`` (no ``ray.init`` in the API process)."""
    code, out, err = _ray_status_subprocess(
        gcs_address, auth_token=auth_token, timeout_sec=timeout_sec
    )
    combined = f"{out}\n{err}".lower()
    if "invalidauthtoken" in combined or code != 0:
        return None
    alive = parse_ray_status_active_nodes(out)
    if alive is None:
        return None
    caps = parse_ray_status_resource_capacities(out)
    return {
        "alive": float(alive),
        "gpu": float(caps.get("gpu", 0.0)),
        "cpu": float(caps.get("cpu", 0.0)),
    }


def ray_cluster_topology_subprocess(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 35.0,
) -> dict[str, Any] | None:
    """
    Full cluster topology inputs without ``ray.init`` in the API / job-runner process.

    Returns alive ``ray.nodes()`` payloads plus cluster resource totals when GCS RPC works.
    """
    from app.services.ray_driver_connect import (
        co_located_subprocess_ray_init_kwargs,
        ensure_ray_auth_token,
    )

    token = (auth_token or ensure_ray_auth_token(gcs_address) or "").strip()
    init_kw = co_located_subprocess_ray_init_kwargs(gcs_address)
    init_attempts: list[dict[str, Any]] = [
        {"address": gcs_address, "ignore_reinit_error": True},
    ]
    if init_kw not in init_attempts:
        init_attempts.append(init_kw)
    script = f"""
import json
import os
import ray

os.environ["RAY_ADDRESS"] = {json.dumps(gcs_address)}
if {json.dumps(token)}:
    os.environ["RAY_AUTH_TOKEN"] = {json.dumps(token)}
attempts = {json.dumps(init_attempts)}
for kwargs in attempts:
    try:
        ray.init(**{{**kwargs, "logging_level": "error"}})
        break
    except Exception:
        if ray.is_initialized():
            ray.shutdown()
else:
    raise SystemExit(1)
nodes = []
for n in ray.nodes():
    if not n.get("Alive"):
        continue
    nodes.append({{
        "NodeID": n.get("NodeID"),
        "NodeManagerAddress": n.get("NodeManagerAddress"),
        "node_ip_address": n.get("node_ip_address"),
        "Resources": dict(n.get("Resources") or {{}}),
        "Alive": True,
    }})
total = {{str(k): float(v) for k, v in ray.cluster_resources().items()}}
avail = {{str(k): float(v) for k, v in ray.available_resources().items()}}
print(json.dumps({{"nodes": nodes, "total_res": total, "avail_res": avail}}))
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
        return data
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def cluster_snapshot_for_wait(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    init_timeout_sec: float = _CLUSTER_SNAPSHOT_TIMEOUT_SEC,
) -> dict[str, float] | None:
    """
    Cluster resource view for provisioning waits.

    Prefer a subprocess ``ray.init`` snapshot when it works; fall back to ``ray status``
    (matches ``ray status`` CLI on the session when co-located ``ray.init`` is flaky).
    """
    snap = ray_cluster_snapshot_subprocess(
        gcs_address, auth_token=auth_token, timeout_sec=init_timeout_sec
    )
    if snap is not None:
        return {**snap, "via": "ray_init"}
    status = ray_status_cluster_snapshot(gcs_address, auth_token=auth_token)
    if status is not None:
        return {**status, "via": "ray_status"}
    return None


def ray_status_cluster_hint(
    gcs_address: str,
    *,
    auth_token: str | None = None,
    timeout_sec: float = 20.0,
) -> tuple[int | None, str]:
    """
    Lightweight cluster view without ``ray.init`` (for UX heartbeats while workers join).
    """
    code, out, err = _ray_status_subprocess(
        gcs_address, auth_token=auth_token, timeout_sec=timeout_sec
    )
    combined = f"{out}\n{err}".lower()
    if "invalidauthtoken" in combined:
        return None, "ray status: InvalidAuthToken — re-run scripts/start.py on the session"
    if code != 0:
        tail = (err or out).strip()[-200:]
        return None, f"ray status failed (exit={code}){': ' + tail if tail else ''}"
    alive = parse_ray_status_active_nodes(out)
    if alive is None:
        return None, "could not parse ray status output"
    caps = parse_ray_status_resource_capacities(out)
    gpu = caps.get("gpu", 0.0)
    return alive, f"ray status: {alive} active node(s), {gpu:g} GPU capacity"


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
    init_attempts: list[dict[str, Any]] = [
        {"address": gcs_address, "ignore_reinit_error": True},
    ]
    if init_kw not in init_attempts:
        init_attempts.append(init_kw)
    script = f"""
import json
import os
import ray

os.environ["RAY_ADDRESS"] = {json.dumps(gcs_address)}
if {json.dumps(token)}:
    os.environ["RAY_AUTH_TOKEN"] = {json.dumps(token)}
attempts = {json.dumps(init_attempts)}
last_err = None
for kwargs in attempts:
    try:
        ray.init(**kwargs)
        break
    except Exception as exc:
        last_err = exc
        if ray.is_initialized():
            ray.shutdown()
else:
    raise SystemExit(1)
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
    poll_sec: float = _CLUSTER_WAIT_POLL_SEC,
    auth_token: str | None = None,
) -> int:
    """Poll Ray GCS until ``min_alive`` nodes (and optional GPUs) are available."""
    seen_counts: set[int] = set()
    started = time.monotonic()
    deadline = started + timeout_sec
    last_heartbeat = started
    need_gpu = min_gpu > 0
    log(
        framework_line(
            "ray",
            f"Waiting for Ray cluster at {gcs_address} (need >={min_alive} node(s)"
            + (f", >={min_gpu:g} GPU" if need_gpu else "")
            + f", timeout {int(timeout_sec)}s)…",
        )
    )
    if need_gpu:
        log(
            framework_line(
                "info",
                "Session Ray head is usually CPU-only; GPU ranks need a CDSW worker "
                "that finished `ray start --block` on the master address.",
            )
        )
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_heartbeat >= _CLUSTER_WAIT_HEARTBEAT_SEC:
            last_heartbeat = now
            elapsed = int(now - started)
            status_alive, status_hint = ray_status_cluster_hint(
                gcs_address, auth_token=auth_token, timeout_sec=20.0
            )
            extra = status_hint
            if status_alive is not None and status_alive < min_alive:
                extra += (
                    f"; waiting for CDSW worker(s) to join (have {status_alive}, need >={min_alive})"
                )
            elif need_gpu and status_alive is not None and status_alive >= min_alive:
                extra += f"; need >={min_gpu:g} GPU in cluster (check worker profile / gpus=1)"
            log(
                framework_line(
                    "ray",
                    f"Still waiting ({elapsed}s): {extra}",
                )
            )
        snap = cluster_snapshot_for_wait(
            gcs_address,
            auth_token=auth_token,
            init_timeout_sec=_CLUSTER_SNAPSHOT_TIMEOUT_SEC,
        )
        if snap is not None:
            count = int(snap["alive"])
            if count not in seen_counts:
                seen_counts.add(count)
                via = str(snap.get("via") or "GCS")
                log(
                    framework_line(
                        "ray",
                        f"Cluster ({via}) reports {count} alive node(s), "
                        f"GPU={snap['gpu']:g} CPU={snap['cpu']:g}",
                    )
                )
            nodes_ok = count >= min_alive
            gpu_ok = not need_gpu or snap["gpu"] >= min_gpu
            if nodes_ok and gpu_ok:
                log(ray_cluster_ready_line(count))
                return count
        else:
            status_alive, status_hint = ray_status_cluster_hint(
                gcs_address, auth_token=auth_token, timeout_sec=20.0
            )
            if status_alive is not None and status_alive not in seen_counts:
                seen_counts.add(status_alive)
                log(framework_line("info", status_hint))
        time.sleep(poll_sec)
    final = cluster_snapshot_for_wait(gcs_address, auth_token=auth_token)
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
