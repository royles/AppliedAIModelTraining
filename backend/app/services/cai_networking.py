"""
Cloudera AI Ray address resolution.

Based on:
- https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-worker-network-communication.html
- https://docs.cloudera.com/machine-learning/cloud/engines/topics/ml-engine-environment-variables.html
"""

from __future__ import annotations

import os

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
