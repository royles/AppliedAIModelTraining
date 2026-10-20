"""Connect the Ray Train driver to a co-located Ray head on CAI."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from app.services.cai_networking import (
    format_ray_head,
    is_cai_master_session,
    master_engine_ip,
    parse_gcs_host_port,
)

logger = logging.getLogger(__name__)

RAY_DRIVER_CONNECT_REVISION = "2026-10-20f"


def _ray_cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "ray.scripts.scripts", *args]


def ensure_ray_auth_token() -> str | None:
    """Load Ray GCS auth token from env, file, or ``ray get-auth-token``."""
    apply_ray_auth_from_session()
    _sync_auth_from_cli()
    token = os.environ.get("RAY_AUTH_TOKEN", "").strip()
    return token or None


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


def _ray_status_ok(gcs_address: str, timeout_sec: float = 120.0) -> bool:
    env = os.environ.copy()
    env["RAY_ADDRESS"] = gcs_address
    apply_ray_auth_from_session()
    if os.environ.get("RAY_AUTH_TOKEN"):
        env["RAY_AUTH_TOKEN"] = os.environ["RAY_AUTH_TOKEN"]
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            proc = subprocess.run(
                _ray_cli("status"),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            time.sleep(2.0)
            continue
        if proc.returncode == 0 and "active" in (proc.stdout or "").lower():
            return True
        time.sleep(2.0)
    return False


def restart_co_located_ray_head(
    *,
    framework_log: Callable[[str], None] | None = None,
) -> str | None:
    """
    ``ray stop`` + ``ray start --head`` on the session (session-only Train).

    Fixes stale GCS / raylet state and ``No node info found`` when ``find_node_ids``
    sees dead raylets or the head raylet never registered with GCS.
    """
    ip = master_engine_ip()
    if not ip:
        return None

    def _log(msg: str) -> None:
        if framework_log:
            framework_log(msg)
        logger.info(msg)

    _log("Restarting co-located Ray head (ray stop --force, then ray start --head)…")
    subprocess.run(
        _ray_cli("stop", "--force"),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    from app.resource_hints import ray_head_start_memory_args
    from app.services.ray_head_options import ray_head_dashboard_cli_args

    args = [
        "start",
        "--head",
        "--port=6379",
        f"--node-ip-address={ip}",
        "--disable-usage-stats",
    ]
    args.extend(ray_head_start_memory_args())
    args.extend(ray_head_dashboard_cli_args())

    proc = subprocess.run(
        _ray_cli(*args),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise RuntimeError(f"ray start --head failed (exit {proc.returncode}): {tail}")

    gcs = format_ray_head(ip)
    os.environ["RAY_ADDRESS"] = gcs
    os.environ["RAY_HEAD_ADDRESS"] = gcs
    _sync_auth_from_cli()
    if not _ray_status_ok(gcs, timeout_sec=90.0):
        raise RuntimeError(f"Ray head at {gcs} did not become active after restart")
    _log(f"Ray head active at {gcs}")
    return gcs


def _sync_auth_from_cli() -> None:
    try:
        proc = subprocess.run(
            _ray_cli("get-auth-token"),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return
    if proc.returncode != 0:
        return
    for line in reversed((proc.stdout or "").splitlines()):
        token = line.strip()
        if token and not token.startswith("Usage"):
            os.environ["RAY_AUTH_TOKEN"] = token
            return


def ensure_session_ray_head(
    *,
    session_only: bool,
    framework_log: Callable[[str], None] | None = None,
) -> None:
    """Before Train on the session pod alone, ensure a single healthy Ray head."""
    if not session_only or not is_cai_master_session():
        return
    gcs = resolve_driver_gcs_address(os.environ.get("RAY_ADDRESS"))
    if gcs and _ray_status_ok(gcs, timeout_sec=15.0):
        return
    restart_co_located_ray_head(framework_log=framework_log)


def _init_attempts(connect: str, master: str | None) -> list[dict[str, Any]]:
    """
    Build ray.init kwargs. Never pass ``_temp_dir`` — it filters GCS nodes and
    causes ``No node info found matching attributes: '<ip>, /tmp/ray'``.
    """
    attempts: list[dict[str, Any]] = [
        {"address": "auto", "ignore_reinit_error": True},
        {"address": connect, "ignore_reinit_error": True},
    ]
    if master:
        attempts.append(
            {
                "address": connect,
                "_node_ip_address": master,
                "ignore_reinit_error": True,
            }
        )
    if _tcp_open("127.0.0.1", 10001):
        attempts.append({"address": "ray://127.0.0.1:10001", "ignore_reinit_error": True})
    return attempts


def _try_ray_init(
    attempts: list[dict[str, Any]],
    *,
    connect: str,
    framework_log: Callable[[str], None] | None,
) -> str:
    import ray
    from ray._private import services

    def _log(msg: str) -> None:
        if framework_log:
            framework_log(msg)
        logger.info(msg)

    last_exc: Exception | None = None
    for kwargs in attempts:
        addr = kwargs.get("address", connect)
        _log(
            f"ray.init attempt address={addr} "
            f"node_ip={kwargs.get('_node_ip_address', 'n/a')}"
        )
        try:
            ray.init(**kwargs)
            _log(f"ray.init OK (address={addr})")
            if addr == "auto":
                return ray_bootstrap_canonical(connect)
            return ray_bootstrap_canonical(str(addr))
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


def init_ray_train_driver(
    ray_address: str | None,
    *,
    worker_head: str | None = None,
    session_only: bool = False,
    framework_log: Callable[[str], None] | None = None,
) -> str:
    """Initialize Ray for the in-process Train driver."""
    import ray
    from ray._private import services

    if ray.is_initialized():
        return ray_address or "local"

    def _log(msg: str) -> None:
        if framework_log:
            framework_log(msg)
        logger.info(msg)

    _log(f"Ray driver connect {_driver_connect_diagnostics(worker_head)}")

    apply_ray_auth_from_session()
    services.find_gcs_addresses.cache_clear()

    connect = resolve_driver_gcs_address(ray_address, worker_head=worker_head)
    if not connect:
        _log("Starting embedded Ray runtime (no cluster address)")
        ray.init(ignore_reinit_error=True)
        return "local"

    master = master_ip_from_context(worker_head)
    os.environ["RAY_ADDRESS"] = connect
    attempts = _init_attempts(connect, master)

    try:
        return _try_ray_init(attempts, connect=connect, framework_log=framework_log)
    except Exception as first_err:  # noqa: BLE001
        if not session_only:
            raise
        _log(f"Driver connect failed ({first_err}); restarting session Ray head…")
        restart_co_located_ray_head(framework_log=framework_log)
        services.find_gcs_addresses.cache_clear()
        connect = resolve_driver_gcs_address(ray_address, worker_head=worker_head) or connect
        os.environ["RAY_ADDRESS"] = connect
        master = master_ip_from_context(worker_head)
        return _try_ray_init(
            _init_attempts(connect, master),
            connect=connect,
            framework_log=framework_log,
        )


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
