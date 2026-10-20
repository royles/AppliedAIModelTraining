"""
Cloudera AI Ray address resolution.

Based on:
- https://docs.cloudera.com/cdsw/1.10.5/parallel-computing/topics/cdsw-example--worker-network-communications.html
- https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-worker-network-communication.html
- https://docs.cloudera.com/machine-learning/cloud/engines/topics/ml-engine-environment-variables.html
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

RAY_GCS_PORT = 6379
RAY_GCS_PORT_MIN = 1024
RAY_GCS_PORT_MAX = 65535


def resolve_ray_gcs_port(address: str | None = None) -> int:
    """GCS port for Ray head and worker join (default 6379; override via ``PLATFORM_RAY_GCS_PORT``)."""
    if address:
        _, port = parse_gcs_host_port(address)
        return port
    raw = os.environ.get("PLATFORM_RAY_GCS_PORT", "").strip()
    port = int(raw) if raw else RAY_GCS_PORT
    if not RAY_GCS_PORT_MIN <= port <= RAY_GCS_PORT_MAX:
        raise ValueError(
            f"PLATFORM_RAY_GCS_PORT must be {RAY_GCS_PORT_MIN}-{RAY_GCS_PORT_MAX}, got {port}"
        )
    return port


def gcs_tcp_reachable(
    host: str,
    port: int = RAY_GCS_PORT,
    *,
    timeout_sec: float = 2.0,
) -> bool:
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def master_engine_ip() -> str | None:
    """IP of this engine as reachable by other CAI engines (master session)."""
    ip = os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")
    if ip:
        return ip.strip()
    head = os.environ.get("RAY_HEAD_ADDRESS") or ""
    if head and ":" in head:
        return head.partition(":")[0].strip() or None
    if head:
        return head.strip()
    return None


def format_ray_head(host: str, port: int | None = None) -> str:
    return f"{host}:{port if port is not None else resolve_ray_gcs_port()}"


def parse_gcs_host_port(address: str) -> tuple[str, int]:
    """Parse a GCS ``host:port`` or ``ray://host:port`` string."""
    raw = address.strip()
    if raw.startswith("ray://"):
        raw = raw[6:]
    host, _, port_s = raw.partition(":")
    if not host:
        host = "127.0.0.1"
    try:
        port = int(port_s or str(RAY_GCS_PORT))
    except ValueError:
        port = RAY_GCS_PORT
    return host, port


def is_cai_master_session() -> bool:
    """
    True when this process runs on the CAI **master session** engine (Ray head host).

    ``cdsw`` / ``cml.workers_v1`` may be missing from the app venv even on CAI; use
    engine env vars and the standard ``/home/cdsw`` session layout as signals.
    """
    if master_engine_ip():
        return True
    home = os.environ.get("HOME", "")
    if home.startswith("/home/cdsw"):
        return True
    return Path("/home/cdsw").is_dir() and os.environ.get("USER", "") == "cdsw"


def coerce_session_driver_ray_address(address: str | None) -> str | None:
    """
    Normalize driver GCS address on the CAI master session.

    Ray 2.x on Linux rewrites ``127.0.0.1`` to the pod IP during ``ray.init`` anyway
    (``resolve_ip_for_localhost``). Use the master engine IP explicitly with
    ``_node_ip_address`` in ``ray_driver_connect.init_ray_train_driver``.
    """
    if not is_cai_master_session():
        return address
    master = master_engine_ip()
    if not master:
        return address
    port = resolve_ray_gcs_port(address)
    return format_ray_head(master, port)


def apply_master_session_ray_driver_env() -> str | None:
    """Set ``RAY_ADDRESS`` for the co-located Ray driver when on a CAI master session."""
    if not is_cai_master_session():
        return None
    addr = coerce_session_driver_ray_address(os.environ.get("RAY_ADDRESS"))
    if not addr:
        addr = ray_driver_gcs_address(os.environ.get("RAY_HEAD_ADDRESS"))
    os.environ["RAY_ADDRESS"] = addr
    return addr


def ray_worker_head_address(explicit: str | None = None) -> str | None:
    """
    Address worker engines use for `ray start --block --address=...`.

    Cloudera Ray examples use the master session's CDSW_IP_ADDRESS on port 6379.
    """
    if explicit:
        return explicit
    ip = master_engine_ip()
    if ip:
        return format_ray_head(ip, resolve_ray_gcs_port())
    return os.environ.get("RAY_HEAD_ADDRESS")


def ray_driver_gcs_address(worker_head: str | None = None) -> str | None:
    """
    GCS address for ``ray.init`` on the master session (same host:port workers use).

    Pair with ``_node_ip_address=master_engine_ip()`` when connecting the driver.
    """
    port = resolve_ray_gcs_port(worker_head)
    master = master_engine_ip()
    if master:
        return format_ray_head(master, port)
    if worker_head:
        return worker_head
    return os.environ.get("RAY_ADDRESS")


def resolve_worker_ray_head_address_with_source(fallback_head: str) -> tuple[str, str]:
    """
    GCS ``host:port`` for CDSW workers and which env key selected it.

    Cloudera workers connect to the **master** using ``CDSW_MASTER_IP`` (see CDSW worker
    network communications). The session pins ``PLATFORM_RAY_HEAD_ADDRESS`` at
    ``launch_workers``; if CAI's ``CDSW_MASTER_IP`` disagrees, use the launch pin.
    """
    pinned = os.environ.get("PLATFORM_RAY_HEAD_ADDRESS", "").strip() or fallback_head.strip()
    pin_host, pin_port = parse_gcs_host_port(pinned) if pinned else ("", resolve_ray_gcs_port())
    port = pin_port if pinned else resolve_ray_gcs_port()
    for key in ("CDSW_MASTER_IP", "CML_MASTER_IP"):
        ip = os.environ.get(key, "").strip()
        if not ip:
            continue
        if pin_host and ip != pin_host:
            break
        return format_ray_head(ip, port), key
    if pinned:
        addr = pinned if ":" in pinned else format_ray_head(pin_host or pinned, port)
        return addr, "PLATFORM_RAY_HEAD_ADDRESS"
    return fallback_head.strip(), "FALLBACK_HEAD"


def resolve_worker_ray_head_address(fallback_head: str) -> str:
    return resolve_worker_ray_head_address_with_source(fallback_head)[0]


def worker_ray_head_from_env(fallback_head: str) -> str:
    return resolve_worker_ray_head_address(fallback_head)


def assert_master_gcs_worker_routable(*, timeout_sec: float = 2.0) -> None:
    """
    Master Ray GCS must accept TCP on ``CDSW_IP_ADDRESS`` (not only loopback).

    Mirrors Cloudera worker networking: master listens on ``0.0.0.0``; workers dial
    ``CDSW_MASTER_IP:<port>``.
    """
    ip = master_engine_ip()
    port = resolve_ray_gcs_port()
    if not ip:
        return
    if gcs_tcp_reachable(ip, port, timeout_sec=timeout_sec):
        return
    if gcs_tcp_reachable("127.0.0.1", port, timeout_sec=timeout_sec):
        raise RuntimeError(
            f"Ray GCS accepts 127.0.0.1:{port} but not master engine {ip}:{port}. "
            "CDSW workers connect via CDSW_MASTER_IP; restart the head on the pod IP "
            "(`ray stop --force`, `python3 scripts/start.py`)."
        )
    raise RuntimeError(
        f"Ray GCS is not listening on master engine {ip}:{port} or 127.0.0.1:{port}. "
        "Run `ray stop --force` and `python3 scripts/start.py`."
    )


def reconcile_live_ray_head_env() -> str | None:
    """Align ``RAY_HEAD_ADDRESS`` / ``RAY_ADDRESS`` with ``CDSW_IP_ADDRESS`` on the session."""
    ip = master_engine_ip()
    if not ip:
        return None
    live = format_ray_head(ip, resolve_ray_gcs_port())
    stale = os.environ.get("RAY_HEAD_ADDRESS", "").strip()
    if stale and stale != live:
        import logging

        logging.getLogger(__name__).warning(
            "RAY_HEAD_ADDRESS %s does not match live session %s; updating env",
            stale,
            live,
        )
    os.environ["RAY_HEAD_ADDRESS"] = live
    return live
