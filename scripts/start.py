#!/usr/bin/env python3
"""
Start platform services (API + UI on CDSW_APP_PORT).

On Cloudera AI / CDSW sessions this also:
  - validates CDSW_APP_PORT
  - runs install.py (unless --skip-install)
  - starts a Ray head on the session pod using the project venv
  - sets RAY_LOCAL_MODE=false

Uses tmux when available (Cloud Agent VMs); otherwise background processes
under /tmp/platform-services/ (typical CDSW).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    REPO_ROOT,
    _log_file,
    cai_session_ip,
    cd_repo,
    is_cai_session,
    start_api_service,
    start_ray_head,
    start_web_service,
    ui_port,
    venv_python,
    wait_for_urls,
)
from install import main as install_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Applied AI Model Training services")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency install (default: install on CAI sessions only)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Force dependency install before start",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cd_repo()
    os.environ.setdefault("CDSW_APP_PORT", ui_port())
    port = ui_port()

    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)

    if is_cai_session():
        if not os.environ.get("CDSW_APP_PORT"):
            print(
                "CDSW_APP_PORT must be set in a Cloudera AI Application / workbench session.",
                file=sys.stderr,
            )
            return 1
        os.environ["RAY_LOCAL_MODE"] = "false"

    should_install = args.install or (is_cai_session() and not args.skip_install)
    if should_install:
        code = install_main()
        if code != 0:
            return code

    if is_cai_session():
        start_ray_head()

    try:
        start_api_service()
        start_web_service(port)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    api_url = "http://127.0.0.1:8000/api/health"
    ui_url = f"http://127.0.0.1:{port}/"
    if wait_for_urls([api_url, ui_url]):
        print(f"API and UI are up (UI port CDSW_APP_PORT={port}).")
        if is_cai_session():
            ip = cai_session_ip()
            print(f"CAI session IP: {ip}")
            print("API: http://127.0.0.1:8000/docs")
            print(f"UI:  use your CAI Application route (listens on CDSW_APP_PORT={port})")
            py = venv_python()
            print(f"Smoke test: {py} {REPO_ROOT / 'scripts' / 'cai_smoke_test.py'}")
        return 0

    print(
        f"Timed out waiting for services; check {_log_file('api')} and {_log_file('web')}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
