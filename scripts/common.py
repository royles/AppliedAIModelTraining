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
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
TMUX_CONF = "/exec-daemon/tmux.portal.conf"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
VENV_PIP = REPO_ROOT / ".venv" / "bin" / "pip"


def cd_repo() -> None:
    os.chdir(REPO_ROOT)


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


def tmux_cmd(*args: str) -> list[str]:
    return ["tmux", "-f", TMUX_CONF, *args]


def tmux_has_session(name: str) -> bool:
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


def http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (HTTPError, URLError, TimeoutError):
        return False


def wait_for_urls(urls: Sequence[str], attempts: int = 60, delay: float = 1.0) -> bool:
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


def ensure_python_venv() -> None:
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

    if not VENV_PYTHON.parent.is_dir():
        run([sys.executable, "-m", "venv", str(REPO_ROOT / ".venv")], check=True)
