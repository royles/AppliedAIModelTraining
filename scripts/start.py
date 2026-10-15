#!/usr/bin/env python3
"""
Start the platform.

Default (CAI and local): FastAPI on CDSW_APP_PORT (CAI) or :8000 (local) serves
the **built** React app from frontend/dist — no npm at runtime.

Optional --dev-ui: Vite dev server (npm required) on CDSW_APP_PORT with API on :8000.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    REPO_ROOT,
    api_health_url,
    cai_session_ip,
    cd_repo,
    ensure_api_running,
    frontend_dist_ready,
    is_cai_session,
    log_file,
    run_api_foreground,
    run_vite_dev_foreground,
    start_ray_head,
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
        help="Run install.py before starting",
    )
    parser.add_argument(
        "--dev-ui",
        action="store_true",
        help="Use Vite dev server (npm required) instead of built frontend/dist",
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

    if is_cai_session():
        os.environ["RAY_LOCAL_MODE"] = "false"
        start_ray_head()

    if args.dev_ui:
        print("=== Dev UI mode (Vite + API on :8000) ===")
        ensure_api_running(8000)
        if not wait_for_urls([api_health_url(8000)], attempts=60):
            print(f"API not ready; see {log_file('api')}", file=sys.stderr)
            return 1
        try:
            return run_vite_dev_foreground(port)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    if not frontend_dist_ready():
        print(
            "frontend/dist is missing. Run: python3 scripts/install.py\n"
            "(needs npm once to build), or pass --dev-ui if npm is available.",
            file=sys.stderr,
        )
        return 1

    print("=== Starting API + built React UI (no npm) ===")
    if is_cai_session():
        ip = cai_session_ip()
        print(f"CAI session IP: {ip}")
        print(f"Application listens on CDSW_APP_PORT={port}")
        print(
            f"Smoke test (other terminal): "
            f"API_BASE=http://127.0.0.1:{port} {venv_python()} {REPO_ROOT / 'scripts' / 'cai_smoke_test.py'}"
        )
        return run_api_foreground(port)

    ensure_api_running(8000)
    if wait_for_urls([api_health_url(8000)], attempts=60):
        print("API + UI available at http://127.0.0.1:8000/ (static build)")
        return 0
    print(f"API not ready; see {log_file('api')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
