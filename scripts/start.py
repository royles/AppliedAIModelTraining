#!/usr/bin/env python3
"""
Start platform services (API + UI on CDSW_APP_PORT).

Install dependencies separately: python3 scripts/install.py
Use --install only if you want start.py to run install first.

On Cloudera AI (CAI), the UI runs in the **foreground** on CDSW_APP_PORT so the
Application route stays alive. API runs in the background on port 8000.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    REPO_ROOT,
    API_HEALTH_URL,
    cai_session_ip,
    cd_repo,
    ensure_api_running,
    is_cai_session,
    log_file,
    run_web_foreground,
    start_ray_head,
    start_web_background,
    ui_health_url,
    ui_port,
    venv_python,
    wait_for_urls,
)
from install import main as install_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Applied AI Model Training services")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run install.py before starting (normally run install separately)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cd_repo()

    if is_cai_session() and not os.environ.get("CDSW_APP_PORT"):
        print(
            "CDSW_APP_PORT must be set in a Cloudera AI Application / workbench session.",
            file=sys.stderr,
        )
        return 1

    os.environ.setdefault("CDSW_APP_PORT", ui_port())
    port = ui_port()

    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)

    if args.install:
        print("=== Installing dependencies ===")
        code = install_main()
        if code != 0:
            return code
        print("=== Install finished; starting services ===")

    if is_cai_session():
        os.environ["RAY_LOCAL_MODE"] = "false"
        start_ray_head()

    print("=== Starting services ===")
    try:
        ensure_api_running()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not wait_for_urls([API_HEALTH_URL], attempts=60):
        print(f"API did not become ready; see {log_file('api')}", file=sys.stderr)
        return 1
    print("API is ready.")

    if is_cai_session():
        ip = cai_session_ip()
        print(f"CAI session IP: {ip}")
        print(f"Open your Application URL (proxies CDSW_APP_PORT={port})")
        print(f"Smoke test (other terminal): {venv_python()} {REPO_ROOT / 'scripts' / 'cai_smoke_test.py'}")
        try:
            return run_web_foreground(port)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    try:
        start_web_background(port)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if wait_for_urls([API_HEALTH_URL, ui_health_url(port)]):
        print(f"API and UI are up (UI port CDSW_APP_PORT={port}).")
        return 0

    print(
        f"Timed out waiting for UI; check {log_file('api')} and {log_file('web')}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
