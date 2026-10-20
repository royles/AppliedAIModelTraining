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
    assert_master_gcs_worker_routable,
    format_ray_head,
    gcs_tcp_reachable,
    master_engine_ip,
    parse_gcs_host_port,
    resolve_ray_gcs_port,
    resolve_worker_ray_head_address,
)

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def live_master_gcs_address() -> str | None:
    ip = master_engine_ip()
    return format_ray_head(ip) if ip else None


def worker_ray_join_address(fallback_head: str) -> str:
    """Address for ``ray start --block --address=...`` on a CDSW worker."""
    return resolve_worker_ray_head_address(fallback_head)


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
        assert_master_gcs_worker_routable()
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
    assert_master_gcs_worker_routable()
    _log(f"Ray head ready at {gcs} (workers join via CDSW_MASTER_IP:{resolve_ray_gcs_port()})")
    return gcs


def describe_worker_probe_failure(probe: dict, head: str) -> str:
    """Human-readable failure when TCP or Ray client checks fail on a CDSW worker."""
    if probe.get("error"):
        return (
            f"CDSW worker connectivity probe timed out ({probe.get('error')}). "
            f"Check CDSW worker logs; Ray head={head}."
        )
    if not probe.get("tcp"):
        return (
            f"CDSW worker cannot open TCP to Ray GCS on the master (head={head}). "
            f"Probe: {probe}. This is a CAI engine networking issue (port 6379 master←worker). "
            "Run `ray stop --force` and `python3 scripts/start.py`, then retry."
        )
    if probe.get("ray_ok"):
        return ""

    hints: list[str] = []
    if not probe.get("auth_configured"):
        hints.append(
            "Ray GCS auth token missing on the worker (set RAY_AUTH_TOKEN on the session or "
            "re-run after `python3 scripts/start.py` so the platform can embed the token)."
        )
    if probe.get("ray_status_ok") is False and probe.get("auth_configured"):
        hints.append(
            "TCP to :6379 works but `ray status` failed — likely Ray version mismatch between "
            "session head and worker Python, or wrong auth token."
        )
    py = probe.get("python")
    if py and ".venv" not in str(py):
        hints.append(
            f"Worker used non-project Python ({py}); expected project .venv Ray matching the head."
        )
    hint_text = " ".join(hints) if hints else (
        "Per Cloudera worker networking, TCP to CDSW_MASTER_IP:6379 means engine routing is OK; "
        "restart the Ray head on the master pod IP (`ray stop --force`, `python3 scripts/start.py`) "
        "and ensure worker Ray matches the session (.venv + RAY_AUTH_TOKEN)."
    )
    return (
        f"CDSW worker cannot join Ray on the master from the worker network (head={head}). "
        f"Probe: {probe}. {hint_text}"
    )


def build_worker_connectivity_probe_code(
    head_address: str,
    result_path: str,
    *,
    ray_auth_token: str | None = None,
) -> str:
    """CDSW worker bootstrap: verify TCP + ``ray status`` + ray.init from worker network."""
    from app.services.cdsw_worker_codegen import embedded_worker_resolve_head_source

    token_json = json.dumps((ray_auth_token or "").strip())
    resolve_head_src = embedded_worker_resolve_head_source()
    default_port = resolve_ray_gcs_port(head_address)
    return f'''import json
import os
import socket
import subprocess
import sys
from pathlib import Path

FALLBACK_HEAD = "{head_address}"
RESULT_PATH = "{result_path}"
INLINE_RAY_AUTH_TOKEN = {token_json}

{resolve_head_src}

def project_root() -> Path:
    for key in ("CDSW_PROJECT_ROOT", "PROJECT_ROOT", "CDSW_WORKING_DIRECTORY"):
        val = os.environ.get(key)
        if val:
            p = Path(val)
            if p.is_dir():
                return p.resolve()
    return Path.cwd().resolve()

def ray_importable(py: str) -> bool:
    return (
        subprocess.call(
            [py, "-c", "import ray"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )

def ensure_ray(py: str, root: Path) -> None:
    if ray_importable(py):
        return
    req = root / "backend" / "requirements.txt"
    env = {{**os.environ, "PIP_USER": "0"}}
    if req.is_file():
        subprocess.check_call([py, "-m", "pip", "install", "-q", "-r", str(req)], env=env, cwd=str(root))
    else:
        subprocess.check_call([py, "-m", "pip", "install", "-q", "ray"], env=env)

def pick_python(root: Path) -> str:
    venv_py = root / ".venv" / "bin" / "python"
    candidates: list[str] = []
    if venv_py.is_file():
        candidates.append(str(venv_py))
    if sys.executable not in candidates:
        candidates.append(sys.executable)
    for candidate in candidates:
        if ray_importable(candidate):
            return candidate
    return candidates[0]

def apply_auth(out: dict) -> None:
    token = INLINE_RAY_AUTH_TOKEN.strip()
    if not token:
        token = os.environ.get("RAY_AUTH_TOKEN", "").strip()
    if not token:
        path = os.environ.get("RAY_AUTH_TOKEN_PATH", "").strip()
        if path and os.path.isfile(path):
            token = Path(path).read_text(encoding="utf-8").strip()
    if token:
        os.environ["RAY_AUTH_TOKEN"] = token
    out["auth_configured"] = bool(token)

def main():
    root = project_root()
    os.chdir(root)
    py = pick_python(root)
    ensure_ray(py, root)
    head = resolve_head()
    out = {{
        "head": head,
        "tcp": False,
        "ray_ok": False,
        "python": py,
        "env": {{k: os.environ.get(k) for k in (
            "CDSW_MASTER_IP", "CML_MASTER_IP", "PLATFORM_RAY_HEAD_ADDRESS", "CDSW_IP_ADDRESS"
        ) if os.environ.get(k)}},
    }}
    apply_auth(out)
    host, _, ps = head.partition(":")
    port = int(ps or "{default_port}")
    try:
        socket.create_connection((host, port), timeout=15).close()
        out["tcp"] = True
    except OSError as exc:
        out["tcp_error"] = str(exc)
        _write(out)
        sys.exit(1)
    ray_env = {{**os.environ, "RAY_ADDRESS": head}}
    if os.environ.get("RAY_AUTH_TOKEN"):
        ray_env["RAY_AUTH_TOKEN"] = os.environ["RAY_AUTH_TOKEN"]
    try:
        proc = subprocess.run(
            [py, "-m", "ray.scripts.scripts", "status"],
            env=ray_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        out["ray_status_ok"] = proc.returncode == 0
        tail = (proc.stdout or "") + (proc.stderr or "")
        if tail:
            out["ray_status_tail"] = tail[-800:]
    except Exception as exc:
        out["ray_status_ok"] = False
        out["ray_status_error"] = str(exc)
    try:
        import ray
        out["ray_version"] = getattr(ray, "__version__", "unknown")
        init_kw = {{"address": head, "ignore_reinit_error": True}}
        node_ip = os.environ.get("CDSW_IP_ADDRESS", "").strip() or os.environ.get(
            "CML_IP_ADDRESS", ""
        ).strip()
        if node_ip:
            init_kw["_node_ip_address"] = node_ip
        ray.init(**init_kw)
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
