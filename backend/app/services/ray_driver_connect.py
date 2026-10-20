"""Connect the Ray Train driver to a co-located Ray head on CAI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from app.services.cai_networking import (
    format_ray_head,
    is_cai_master_session,
    master_engine_ip,
    parse_gcs_host_port,
)

logger = logging.getLogger(__name__)


def apply_ray_auth_from_session() -> bool:
    """Load Ray 2.59+ GCS auth token into the driver process if missing."""
    if os.environ.get("RAY_AUTH_TOKEN", "").strip():
        return True
    token_path = os.environ.get("RAY_AUTH_TOKEN_PATH", "").strip()
    candidates: list[Path] = []
    if token_path:
        candidates.append(Path(token_path))
    candidates.append(Path.home() / ".ray" / "auth_token")
    for path in candidates:
        try:
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    os.environ["RAY_AUTH_TOKEN"] = token
                    return True
        except OSError:
            continue
    return False


def ray_bootstrap_canonical(address: str) -> str:
    from ray._private import services

    return services.canonicalize_bootstrap_address(address) or address


def resolve_driver_gcs_address(explicit: str | None) -> str | None:
    """
    GCS address for ``ray.init`` on the master session.

    On Linux Ray 2.x, ``127.0.0.1:6379`` is rewritten to the pod IP via
    ``resolve_ip_for_localhost`` — use the master engine IP explicitly and pass
    ``_node_ip_address`` so the driver matches the head node in GCS.
    """
    master = master_engine_ip()
    if is_cai_master_session() and master:
        port = 6379
        if explicit:
            _, port = parse_gcs_host_port(explicit)
        return format_ray_head(master, port)
    return explicit


def init_ray_train_driver(
    ray_address: str | None,
    *,
    framework_log: Callable[[str], None] | None = None,
) -> str:
    """
    Initialize Ray for the in-process Train driver.

    Returns the canonical GCS address Ray connected to (for job metadata).
    """
    import ray
    from ray._private import services

    if ray.is_initialized():
        return ray_address or "local"

    apply_ray_auth_from_session()
    services.find_gcs_addresses.cache_clear()

    connect = resolve_driver_gcs_address(ray_address)
    if not connect:
        if framework_log:
            framework_log("Starting embedded Ray runtime (no cluster address)")
        ray.init(ignore_reinit_error=True)
        return "local"

    canonical = ray_bootstrap_canonical(connect)
    node_ip = master_engine_ip()
    if node_ip:
        host, _ = parse_gcs_host_port(connect)
        if host in ("127.0.0.1", "localhost"):
            connect = format_ray_head(node_ip)
            canonical = ray_bootstrap_canonical(connect)

    os.environ["RAY_ADDRESS"] = connect
    if framework_log:
        framework_log(
            f"Ray driver GCS connect {connect}"
            + (f" (canonical {canonical})" if canonical != connect else "")
            + (f"; node_ip={node_ip}" if node_ip else "")
        )

    init_kwargs: dict = {"address": connect, "ignore_reinit_error": True}
    if node_ip:
        init_kwargs["_node_ip_address"] = node_ip

    ray.init(**init_kwargs)
    return canonical
