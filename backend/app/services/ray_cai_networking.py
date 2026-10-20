"""
Cloudera AI Ray cluster networking — single place for head + worker join addresses.

Workers must reach GCS on the master engine pod IP (``CDSW_IP_ADDRESS:6379``).
The session driver may use the same address with ``_node_ip_address`` (see
``ray_driver_connect``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable

from app.services.cai_networking import (
    format_ray_head,
    gcs_tcp_reachable,
    master_engine_ip,
    parse_gcs_host_port,
)

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def live_master_gcs_address() -> str | None:
    ip = master_engine_ip()
    return format_ray_head(ip) if ip else None


def worker_ray_join_address(fallback_head: str) -> str:
    """
    Address for ``ray start --block --address=...`` on a CDSW worker.

    Prefer CAI ``CDSW_MASTER_IP`` when it matches the launch-time head; otherwise
    use the launch pin (``PLATFORM_RAY_HEAD_ADDRESS`` / fallback).
    """
    pinned = os.environ.get("PLATFORM_RAY_HEAD_ADDRESS", "").strip() or fallback_head.strip()
    pin_host, _ = parse_gcs_host_port(pinned)
    for key in ("CDSW_MASTER_IP", "CML_MASTER_IP"):
        ip = os.environ.get(key, "").strip()
        if not ip:
            continue
        if ip == pin_host:
            return format_ray_head(ip)
    return pinned


def ensure_ray_head_for_workers(
    *,
    framework_log: LogFn | None = None,
    force_restart: bool | None = None,
) -> str:
    """
    Guarantee Ray GCS is up on the **live** master pod IP (worker-routable).

    Restarts the co-located head when RPC to the pod IP fails (common after
    session reschedule or localhost-only stale raylets).
    """
    gcs = live_master_gcs_address()
    if not gcs:
        raise RuntimeError(
            "CDSW_IP_ADDRESS / CML_IP_ADDRESS is not set; cannot start Ray for workers"
        )

    def _log(msg: str) -> None:
        if framework_log:
            framework_log(msg)
        logger.info(msg)

    if force_restart is None:
        force_restart = os.environ.get("PLATFORM_RAY_FORCE_HEAD_RESTART", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    host, _ = parse_gcs_host_port(gcs)
    from app.services.ray_cluster_watch import wait_for_gcs_rpc_subprocess
    from app.services.ray_driver_connect import ensure_ray_auth_token, restart_co_located_ray_head

    token = ensure_ray_auth_token()

    def _rpc_ready(timeout: float) -> bool:
        return (
            gcs_tcp_reachable(host, timeout_sec=2.0)
            and wait_for_gcs_rpc_subprocess(gcs, auth_token=token, timeout_sec=timeout)
        )

    if force_restart:
        _log("Ensuring fresh Ray head on master pod IP (ray stop --force)…")
        restart_co_located_ray_head(framework_log=framework_log)
    elif _rpc_ready(25.0):
        os.environ["RAY_HEAD_ADDRESS"] = gcs
        os.environ["RAY_ADDRESS"] = gcs
        _log(f"Ray head healthy at {gcs}")
        return gcs

    if not force_restart:
        _log(f"Ray GCS not healthy at {gcs}; restarting head…")
        restart_co_located_ray_head(framework_log=framework_log)

    if not _rpc_ready(90.0):
        raise RuntimeError(
            f"Ray GCS is not reachable at {gcs} from the master session after restart. "
            "CDSW workers cannot join. Run `ray stop --force` and `python3 scripts/start.py`, "
            "then check start-user.log for `ray start --head` errors."
        )

    os.environ["RAY_HEAD_ADDRESS"] = gcs
    os.environ["RAY_ADDRESS"] = gcs
    _log(f"Ray head ready at {gcs} (worker join address)")
    return gcs


def build_worker_connectivity_probe_code(head_address: str, result_path: str) -> str:
    """CDSW worker bootstrap: verify TCP + ray.init from the worker network namespace."""
    return f'''import json
import os
import socket
import sys

HEAD = "{head_address}"
RESULT_PATH = "{result_path}"

def main():
    host, _, ps = HEAD.partition(":")
    port = int(ps or "6379")
    out = {{
        "head": HEAD,
        "tcp": False,
        "ray_ok": False,
        "env": {{k: os.environ.get(k) for k in (
            "CDSW_MASTER_IP", "CML_MASTER_IP", "PLATFORM_RAY_HEAD_ADDRESS", "CDSW_IP_ADDRESS"
        ) if os.environ.get(k)}},
    }}
    try:
        socket.create_connection((host, port), timeout=15).close()
        out["tcp"] = True
    except OSError as exc:
        out["tcp_error"] = str(exc)
        _write(out)
        sys.exit(1)
    try:
        import ray
        token = os.environ.get("RAY_AUTH_TOKEN", "").strip()
        if token:
            os.environ["RAY_AUTH_TOKEN"] = token
        ray.init(address=HEAD, ignore_reinit_error=True)
        out["ray_ok"] = ray.is_initialized()
        out["nodes"] = len(ray.nodes())
        ray.shutdown()
    except Exception as exc:
        out["ray_error"] = str(exc)
        _write(out)
        sys.exit(2)
    _write(out)
    sys.exit(0)

def _write(body):
    text = json.dumps(body)
    print(text, flush=True)
    if RESULT_PATH:
        os.makedirs(os.path.dirname(RESULT_PATH) or ".", exist_ok=True)
        with open(RESULT_PATH, "w", encoding="utf-8") as fp:
            fp.write(text)

if __name__ == "__main__":
    main()
'''


def poll_worker_connectivity_probe(
    result_path: str | Path,
    *,
    timeout_sec: float = 300.0,
    poll_sec: float = 5.0,
) -> dict:
    import time

    path = Path(result_path)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        time.sleep(poll_sec)
    return {"error": f"timeout waiting for {path}"}
