"""
Cloudera AI Ray address resolution.

Based on:
- https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-worker-network-communication.html
- https://docs.cloudera.com/machine-learning/cloud/engines/topics/ml-engine-environment-variables.html
"""

from __future__ import annotations

import os
from pathlib import Path

RAY_GCS_PORT = 6379


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


def format_ray_head(host: str, port: int = RAY_GCS_PORT) -> str:
    return f"{host}:{port}"


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
    Ray Train driver on the master session must use loopback for GCS (port 6379).

    Workers continue to use ``CDSW_IP_ADDRESS:6379``; only the in-pod driver is rewritten.
    """
    if not address:
        if is_cai_master_session():
            return ray_driver_gcs_address(os.environ.get("RAY_HEAD_ADDRESS"))
        return address

    host, port = parse_gcs_host_port(address)
    if host in ("127.0.0.1", "localhost"):
        return format_ray_head("127.0.0.1", port)

    if not is_cai_master_session() or port != RAY_GCS_PORT:
        return address

    # Co-located Ray head on this session — pod IP triggers "No node info found" in ray.init.
    return format_ray_head("127.0.0.1", RAY_GCS_PORT)


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
        return format_ray_head(ip)
    return os.environ.get("RAY_HEAD_ADDRESS")


def ray_driver_gcs_address(worker_head: str | None = None) -> str | None:
    """
    Address for ray.init() / Ray Train driver in the **master session pod**.

    HTTP services in an engine should bind 127.0.0.1 (CDSW_APP_PORT docs); the
    driver uses loopback to reach the co-located Ray head GCS.
    """
    port = RAY_GCS_PORT
    if worker_head and ":" in worker_head:
        try:
            port = int(worker_head.partition(":")[2] or str(RAY_GCS_PORT))
        except ValueError:
            port = RAY_GCS_PORT
    env = os.environ.get("RAY_ADDRESS")
    if env and env.startswith("127.0.0.1"):
        return env
    return format_ray_head("127.0.0.1", port)


def worker_ray_head_from_env(fallback_head: str) -> str:
    """
    Resolve Ray head on a worker engine.

    Workers should prefer CDSW_MASTER_IP / CML_MASTER_IP set by CAI (see worker
    network communication docs), then fall back to the address embedded at launch.
    """
    for key in ("CDSW_MASTER_IP", "CML_MASTER_IP", "CDSW_IP_ADDRESS", "CML_IP_ADDRESS"):
        ip = os.environ.get(key)
        if ip:
            return format_ray_head(ip.strip())
    return fallback_head
