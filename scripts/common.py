"""Shared helpers for platform lifecycle scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
BIND_HOST = "127.0.0.1"
TMUX_CONF = "/exec-daemon/tmux.portal.conf"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_RAY = REPO_ROOT / ".venv" / "bin" / "ray"
RUN_DIR = Path("/tmp/platform-services")
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


def cd_repo() -> None:
    os.chdir(REPO_ROOT)


def frontend_dist_ready() -> bool:
    return FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").is_file()


def cai_session_ip() -> str | None:
    return os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")


def is_cai_session() -> bool:
    return cai_session_ip() is not None


def api_health_url(port: str | int | None = None) -> str:
    p = port if port is not None else api_port()
    return f"http://{BIND_HOST}:{p}/api/health"


def ui_health_url(port: str | int) -> str:
    return f"http://{BIND_HOST}:{port}/"


def resolve_ux_port() -> str:
    """CAI: CDSW_APP_PORT from the environment (read once at startup). Local: PLATFORM_UI_PORT."""
    if is_cai_session():
        port = os.environ.get("CDSW_APP_PORT")
        if not port:
            raise RuntimeError("CDSW_APP_PORT must be set in a CAI session.")
        return port
    return os.environ.get("PLATFORM_UI_PORT", "5173")


def _port_is_open(port: int) -> bool:
    import socket

    try:
        with socket.create_connection((BIND_HOST, port), timeout=1.0):
            return True
    except OSError:
        return False


def hold_forever() -> int:
    """Keep the Application process alive without binding CDSW_APP_PORT again."""
    print("Process will stay running (Ctrl+C to stop).", flush=True)
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        return 0


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        list(cmd),
        cwd=str(cwd or REPO_ROOT),
        env=merged,
        check=check,
        text=True,
    )


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def tmux_cmd(*args: str) -> list[str]:
    if Path(TMUX_CONF).is_file():
        return ["tmux", "-f", TMUX_CONF, *args]
    return ["tmux", *args]


def tmux_has_session(name: str) -> bool:
    if not tmux_available():
        return False
    return (
        subprocess.run(
            tmux_cmd("has-session", "-t", name),
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def tmux_new_session(name: str, cwd: Path, shell_command: str) -> None:
    run(
        tmux_cmd(
            "new-session",
            "-d",
            "-s",
            name,
            "-c",
            str(cwd),
            shell_command,
        )
    )


def log_file(name: str) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{name}.log"


def _pid_file(name: str) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{name}.pid"


def process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def clear_stale_pid(name: str) -> None:
    pid_path = _pid_file(name)
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return
    if not process_running(pid):
        pid_path.unlink(missing_ok=True)


def http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (HTTPError, URLError, TimeoutError):
        return False


def wait_for_urls(urls: Sequence[str], attempts: int = 90, delay: float = 1.0) -> bool:
    for _ in range(attempts):
        if all(http_ok(u) for u in urls):
            return True
        time.sleep(delay)
    return False


def api_port() -> str:
    return os.environ.get("PLATFORM_API_PORT", "8000")


def venv_python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def venv_ray_argv(*ray_args: str) -> list[str]:
    if VENV_RAY.is_file():
        return [str(VENV_RAY), *ray_args]
    raise FileNotFoundError(
        "Ray is not installed in .venv. Run: python3 scripts/install.py"
    )


def venv_subprocess_env() -> dict[str, str]:
    merged = os.environ.copy()
    merged["PIP_USER"] = "0"
    merged["PYTHONNOUSERSITE"] = "1"
    for key in ("PYTHONUSERBASE", "PIP_USER_SITE"):
        merged.pop(key, None)
    venv = REPO_ROOT / ".venv"
    if venv.is_dir():
        merged["VIRTUAL_ENV"] = str(venv)
        bin_dir = venv / "bin"
        merged["PATH"] = f"{bin_dir}{os.pathsep}{merged.get('PATH', '')}"
    return merged


def pip_install(pip_args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    py = venv_python()
    return run(
        [py, "-m", "pip", "install", *pip_args],
        env=venv_subprocess_env(),
        check=check,
    )


def ensure_python_venv() -> None:
    if VENV_PYTHON.is_file():
        return

    probe = REPO_ROOT / ".tmp_venv_probe"
    try:
        run([sys.executable, "-m", "venv", str(probe)], check=True)
    except subprocess.CalledProcessError:
        run(["sudo", "apt-get", "update", "-qq"], check=False)
        run(
            ["sudo", "apt-get", "install", "-y", "-qq", "python3.12-venv"],
            check=True,
        )
    finally:
        shutil.rmtree(probe, ignore_errors=True)

    if not VENV_PYTHON.is_file():
        run([sys.executable, "-m", "venv", str(REPO_ROOT / ".venv")], check=True)


def _ray_port_open(port: int = 6379) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def start_ray_head() -> None:
    ip = cai_session_ip()
    if not ip:
        return
    if os.environ.get("RAY_SKIP_HEAD", "").lower() in ("1", "true", "yes"):
        print("Skipping Ray head startup (RAY_SKIP_HEAD is set).")
        return

    os.environ["RAY_HEAD_ADDRESS"] = f"{ip}:6379"

    if _ray_port_open(6379):
        print("Ray head already listening on :6379")
        return

    include_dashboard = os.environ.get("RAY_INCLUDE_DASHBOARD", "").lower() in (
        "1",
        "true",
        "yes",
    )
    ray_args = ["start", "--head", "--port=6379"]
    if include_dashboard:
        ray_args.extend(["--dashboard-port=8265", "--include-dashboard=true"])
    else:
        # CAI images often lack ray[default] dashboard deps; training only needs the GCS port.
        ray_args.append("--include-dashboard=false")

    print(f"CAI session at {ip}; starting Ray head (dashboard={include_dashboard})...")
    try:
        result = run(
            venv_ray_argv(*ray_args),
            env=venv_subprocess_env(),
            check=False,
        )
    except FileNotFoundError:
        print(
            "Warning: Ray CLI not found in venv; run: python3 scripts/install.py",
            file=sys.stderr,
        )
        return

    if result.returncode != 0:
        print(
            "Warning: Ray head did not start. Training may use local Ray or workers "
            "once the cluster is up. To skip this step: RAY_SKIP_HEAD=1. "
            "Dashboard (optional): RAY_INCLUDE_DASHBOARD=1 after installing ray[default].",
            file=sys.stderr,
        )


def start_background_process(
    name: str,
    *,
    cwd: Path,
    argv: Sequence[str],
    env: dict[str, str] | None = None,
) -> None:
    log_path = log_file(name)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_fp:
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=merged,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _pid_file(name).write_text(str(proc.pid))
    print(f"Started {name} pid={proc.pid} log={log_path}")


def _api_upstream(api: str | int) -> str:
    return f"http://127.0.0.1:{api}"


def _ui_gateway_env(api: str | int) -> dict[str, str]:
    env = venv_subprocess_env()
    env["PLATFORM_API_UPSTREAM"] = _api_upstream(api)
    return env


def run_ui_gateway_foreground(ui: str | int, api: str | int) -> int:
    """Bind UX exactly once on CDSW_APP_PORT (127.0.0.1 only)."""
    ui_s = str(ui)
    health = ui_health_url(ui_s)
    if http_ok(health):
        print(
            f"UX already listening on {BIND_HOST}:{ui_s} (CDSW_APP_PORT); "
            "not starting a second listener."
        )
        return hold_forever()
    if _port_is_open(int(ui_s)):
        if is_cai_session() and http_ok(api_health_url(api)):
            print(
                f"Port {ui_s} (CDSW_APP_PORT) is already in use; "
                f"API is healthy on {BIND_HOST}:{api} — not rebinding UX."
            )
            return hold_forever()
        print(
            f"Port {ui_s} (CDSW_APP_PORT) is already in use by another process.",
            file=sys.stderr,
        )
        return 1

    py = venv_python()
    print(
        f"Starting UX on {BIND_HOST}:{ui_s} (CDSW_APP_PORT; API on {BIND_HOST}:{api}) ..."
    )
    return subprocess.call(
        [
            py,
            "-m",
            "uvicorn",
            "app.ui_gateway:app",
            "--host",
            BIND_HOST,
            "--port",
            ui_s,
        ],
        cwd=str(REPO_ROOT / "backend"),
        env=_ui_gateway_env(api),
    )


def start_ui_gateway_background(ui: str | int, api: str | int) -> None:
    ui_s = str(ui)
    if http_ok(ui_health_url(ui_s)):
        print(f"UX already responding on {BIND_HOST}:{ui_s}")
        return
    if _port_is_open(int(ui_s)):
        print(f"Port {ui_s} already in use; not starting UX.", file=sys.stderr)
        return
    py = venv_python()
    print(f"Starting UX on {BIND_HOST}:{ui_s} ...")
    clear_stale_pid("ui")
    start_background_process(
        "ui",
        cwd=REPO_ROOT / "backend",
        argv=[
            py,
            "-m",
            "uvicorn",
            "app.ui_gateway:app",
            "--host",
            BIND_HOST,
            "--port",
            ui_s,
        ],
        env=_ui_gateway_env(api),
    )


def ensure_api_running(port: str | int | None = None) -> None:
    port = port if port is not None else api_port()
    health = api_health_url(port)
    if http_ok(health):
        print(f"API already responding on :{port}")
        return
    print(f"Starting API on :{port} ...")
    py = venv_python()
    if tmux_available():
        if not tmux_has_session("api"):
            api_cmd = (
                f'"{py}" -m uvicorn app.main:app --host {BIND_HOST} --port {port} '
                f"2>&1 | tee {log_file('api')}"
            )
            tmux_new_session("api", REPO_ROOT / "backend", api_cmd)
        return
    clear_stale_pid("api")
    start_background_process(
        "api",
        cwd=REPO_ROOT / "backend",
        argv=[
            py,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            BIND_HOST,
            "--port",
            str(port),
        ],
        env=venv_subprocess_env(),
    )


def run_vite_dev_foreground(port: str) -> int:
    """Optional hot-reload dev server (requires npm). Binds CDSW_APP_PORT once."""
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required for --dev-ui (Vite dev server)")
    if http_ok(ui_health_url(port)) or _port_is_open(int(port)):
        print(f"Port {port} already in use; not starting Vite.", file=sys.stderr)
        return 1
    env = venv_subprocess_env()
    env["PLATFORM_UI_PORT"] = port
    print(f"Starting Vite dev server on {BIND_HOST}:{port} ...")
    return subprocess.call(
        [npm, "run", "dev"],
        cwd=str(REPO_ROOT / "frontend"),
        env=env,
    )
