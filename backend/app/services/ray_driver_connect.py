"""Connect the Ray Train driver to a co-located Ray head on CAI."""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Any, Callable

from app.services.cai_networking import (
    format_ray_head,
    is_cai_master_session,
    master_engine_ip,
    parse_gcs_host_port,
)

logger = logging.getLogger(__name__)

# Bump when driver connect logic changes — surfaced in /api/health and framework logs.
RAY_DRIVER_CONNECT_REVISION = "2026-10-20c"


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


def master_ip_from_context(worker_head: str | None = None) -> str | None:
    """Master engine IP from CAI env, else parse from worker head ``host:6379``."""
    ip = master_engine_ip()
    if ip:
        return ip
    if worker_head:
        host, _ = parse_gcs_host_port(worker_head)
        if host and host not in ("127.0.0.1", "localhost"):
            return host
    return None


def ray_bootstrap_canonical(address: str) -> str:
    from ray._private import services

    return services.canonicalize_bootstrap_address(address) or address


def resolve_driver_gcs_address(
    explicit: str | None,
    *,
    worker_head: str | None = None,
) -> str | None:
    """
    GCS address for ``ray.init`` on the master session.

    On Linux Ray 2.x, ``127.0.0.1:6379`` is rewritten to the pod IP via
    ``resolve_ip_for_localhost`` — use the master engine IP explicitly and pass
    ``_node_ip_address`` so the driver matches the head node in GCS.
    """
    master = master_ip_from_context(worker_head)
    on_master = is_cai_master_session() or master is not None
    if on_master and master:
        port = 6379
        if explicit:
            _, port = parse_gcs_host_port(explicit)
        elif worker_head:
            _, port = parse_gcs_host_port(worker_head)
        return format_ray_head(master, port)
    return explicit


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _driver_connect_diagnostics(worker_head: str | None) -> str:
    master = master_ip_from_context(worker_head)
    auth = bool(os.environ.get("RAY_AUTH_TOKEN", "").strip()) or (
        Path.home() / ".ray" / "auth_token"
    ).is_file()
    return (
        f"rev={RAY_DRIVER_CONNECT_REVISION} "
        f"cai_master={is_cai_master_session()} "
        f"master_ip={master or 'n/a'} "
        f"CDSW_IP={os.environ.get('CDSW_IP_ADDRESS') or os.environ.get('CML_IP_ADDRESS') or 'n/a'} "
        f"RAY_ADDRESS={os.environ.get('RAY_ADDRESS') or 'n/a'} "
        f"auth={'yes' if auth else 'no'}"
    )


def init_ray_train_driver(
    ray_address: str | None,
    *,
    worker_head: str | None = None,
    framework_log: Callable[[str], None] | None = None,
) -> str:
    """
    Initialize Ray for the in-process Train driver.

    Returns the canonical GCS address Ray connected to (for job metadata).
    """
    import ray
    from ray._private import services

    def _log(msg: str) -> None:
        if framework_log:
            framework_log(msg)
        logger.info(msg)

    if ray.is_initialized():
        return ray_address or "local"

    _log(f"Ray driver connect {_driver_connect_diagnostics(worker_head)}")

    apply_ray_auth_from_session()
    services.find_gcs_addresses.cache_clear()

    connect = resolve_driver_gcs_address(ray_address, worker_head=worker_head)
    if not connect:
        _log("Starting embedded Ray runtime (no cluster address)")
        ray.init(ignore_reinit_error=True)
        return "local"

    master = master_ip_from_context(worker_head)
    canonical = ray_bootstrap_canonical(connect)
    os.environ["RAY_ADDRESS"] = connect

    attempts: list[dict[str, Any]] = []
    if master:
        attempts.append(
            {
                "address": connect,
                "_node_ip_address": master,
                "_temp_dir": os.environ.get("RAY_TMPDIR") or "/tmp/ray",
                "ignore_reinit_error": True,
            }
        )
        attempts.append(
            {
                "address": "auto",
                "_node_ip_address": master,
                "_temp_dir": os.environ.get("RAY_TMPDIR") or "/tmp/ray",
                "ignore_reinit_error": True,
            }
        )
    else:
        attempts.append({"address": connect, "ignore_reinit_error": True})

    if _tcp_open("127.0.0.1", 10001):
        attempts.append({"address": "ray://127.0.0.1:10001", "ignore_reinit_error": True})

    last_exc: Exception | None = None
    for kwargs in attempts:
        addr = kwargs.get("address", connect)
        _log(
            f"ray.init attempt address={addr} "
            f"node_ip={kwargs.get('_node_ip_address', 'n/a')} "
            f"canonical_gcs={canonical}"
        )
        try:
            ray.init(**kwargs)
            _log(f"ray.init OK (address={addr})")
            return ray_bootstrap_canonical(str(addr)) if addr != "auto" else canonical
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            _log(f"ray.init failed: {exc}")
            if ray.is_initialized():
                try:
                    ray.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            services.find_gcs_addresses.cache_clear()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("ray.init failed with no attempts")


def health_ray_driver_info() -> dict[str, str | None]:
    head = os.environ.get("RAY_HEAD_ADDRESS")
    return {
        "ray_driver_connect_revision": RAY_DRIVER_CONNECT_REVISION,
        "master_engine_ip": master_engine_ip(),
        "resolved_driver_gcs": resolve_driver_gcs_address(
            os.environ.get("RAY_ADDRESS"), worker_head=head
        ),
        "ray_auth_configured": "yes"
        if os.environ.get("RAY_AUTH_TOKEN") or (Path.home() / ".ray" / "auth_token").is_file()
        else "no",
    }
