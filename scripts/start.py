#!/usr/bin/env python3
"""Start API and React UI (Vite on CDSW_APP_PORT)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    REPO_ROOT,
    cd_repo,
    tmux_has_session,
    tmux_new_session,
    ui_port,
    venv_python,
    wait_for_urls,
)


def main() -> int:
    cd_repo()
    os.environ.setdefault("CDSW_APP_PORT", ui_port())
    port = ui_port()

    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)

    py = venv_python()
    if not tmux_has_session("api"):
        api_cmd = (
            f'"{py}" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 '
            "2>&1 | tee /tmp/api.log"
        )
        tmux_new_session("api", REPO_ROOT / "backend", api_cmd)

    if not tmux_has_session("web"):
        web_cmd = (
            f'export CDSW_APP_PORT="{port}" && npm run dev 2>&1 | tee /tmp/web.log'
        )
        tmux_new_session("web", REPO_ROOT / "frontend", web_cmd)

    api_url = "http://127.0.0.1:8000/api/health"
    ui_url = f"http://127.0.0.1:{port}/"
    if wait_for_urls([api_url, ui_url]):
        print(f"API and UI are up (UI port CDSW_APP_PORT={port}).")
        return 0

    print(
        "Timed out waiting for services; check /tmp/api.log and /tmp/web.log",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
