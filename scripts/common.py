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
TMUX_CONF = "/exec-daemon/tmux.portal.conf"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_RAY = REPO_ROOT / ".venv" / "bin" / "ray"
RUN_DIR = Path("/tmp/platform-services")

API_HEALTH_URL = "http://127.0.0.1:8000/api/health"


def cd_repo() -> None:
    os.chdir(REPO_ROOT)


def cai_session_ip() -> str | None:
    return os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")


def is_cai_session() -> bool:
    return cai_session_ip() is not None


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


def ui_health_url(port: str) -> str:
    return f"http://127.0.0.1:{port}/"


def wait_for_urls(urls: Sequence[str], attempts: int = 90, delay: float = 1.0) -> bool:
    for _ in range(attempts):
        if all(http_ok(u) for u in urls):
            return True
        time.sleep(delay)
    return False


def ui_port() -> str:
    return os.environ.get("CDSW_APP_PORT", "5173")


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


def start_ray_head() -> None:
    ip = cai_session_ip()
    if not ip:
        return
    os.environ["RAY_HEAD_ADDRESS"] = f"{ip}:6379"
    print(f"CAI session at {ip}; starting Ray head...")
    try:
        run(
            venv_ray_argv(
                "start",
                "--head",
                "--port=6379",
                "--dashboard-port=8265",
                "--include-dashboard=true",
            ),
            env=venv_subprocess_env(),
            check=False,
        )
    except FileNotFoundError:
        print(
            "Warning: Ray CLI not found in venv; run: python3 scripts/install.py",
            file=sys.stderr,
        )


def _start_api_background() -> None:
    py = venv_python()
    clear_stale_pid("api")
    start_background_process(
        "api",
        cwd=REPO_ROOT / "backend",
        argv=[py, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=venv_subprocess_env(),
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


def ensure_api_running() -> None:
    if http_ok(API_HEALTH_URL):
        print("API already responding on :8000")
        return
    print("Starting API on :8000 ...")
    py = venv_python()
    if tmux_available():
        if not tmux_has_session("api"):
            api_cmd = (
                f'"{py}" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 '
                f"2>&1 | tee {log_file('api')}"
            )
            tmux_new_session("api", REPO_ROOT / "backend", api_cmd)
        return
    _start_api_background()


def run_web_foreground(port: str) -> int:
    """Block on Vite — required for CAI Application routes on CDSW_APP_PORT."""
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required to start the UI")
    env = venv_subprocess_env()
    env["CDSW_APP_PORT"] = port
    print(f"Starting UI in foreground on CDSW_APP_PORT={port} ...")
    return subprocess.call(
        [npm, "run", "dev"],
        cwd=str(REPO_ROOT / "frontend"),
        env=env,
    )


def start_web_background(port: str) -> None:
    if http_ok(ui_health_url(port)):
        print(f"UI already responding on CDSW_APP_PORT={port}")
        return
    print(f"Starting UI on CDSW_APP_PORT={port} ...")
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required to start the UI")

    if tmux_available():
        if not tmux_has_session("web"):
            web_cmd = (
                f'export CDSW_APP_PORT="{port}" && npm run dev '
                f"2>&1 | tee {log_file('web')}"
            )
            tmux_new_session("web", REPO_ROOT / "frontend", web_cmd)
        return

    clear_stale_pid("web")
    env = venv_subprocess_env()
    env["CDSW_APP_PORT"] = port
    start_background_process(
        "web",
        cwd=REPO_ROOT / "frontend",
        argv=[npm, "run", "dev"],
        env=env,
    )
