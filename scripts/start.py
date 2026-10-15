#!/usr/bin/env python3
"""
Start the platform (split ports):

- **UX** on CDSW_APP_PORT (CAI Application) — static React + proxy to API
- **API** on PLATFORM_API_PORT (default 8000) — FastAPI, Ray training, SQLite

No npm at runtime when using frontend/dist.
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
    api_port,
    cai_session_ip,
    cd_repo,
    ensure_api_running,
    frontend_dist_ready,
    is_cai_session,
    log_file,
    run_ui_gateway_foreground,
    run_vite_dev_foreground,
    start_ray_head,
    start_ui_gateway_background,
    ui_port,
    venv_python,
    wait_for_urls,
)
from install import main as install_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Applied AI Model Training services")
    parser.add_argument("--install", action="store_true", help="Run install.py first")
    parser.add_argument(
        "--dev-ui",
        action="store_true",
        help="Vite dev server on CDSW_APP_PORT (npm) + API on PLATFORM_API_PORT",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cd_repo()

    if is_cai_session() and not os.environ.get("CDSW_APP_PORT"):
        print("CDSW_APP_PORT must be set for the CAI Application UX.", file=sys.stderr)
        return 1

    os.environ.setdefault("CDSW_APP_PORT", ui_port())
    ux_port = ui_port()
    backend_port = api_port()

    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)

    if args.install:
        code = install_main()
        if code != 0:
            return code

    if is_cai_session():
        os.environ["RAY_LOCAL_MODE"] = "false"
        start_ray_head()

    if args.dev_ui:
        print(f"=== Dev UI: Vite :{ux_port}, API :{backend_port} ===")
        ensure_api_running(backend_port)
        if not wait_for_urls([api_health_url(backend_port)], attempts=60):
            print(f"API not ready; see {log_file('api')}", file=sys.stderr)
            return 1
        try:
            return run_vite_dev_foreground(ux_port)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    if not frontend_dist_ready():
        print(
            "frontend/dist is missing. Run: python3 scripts/install.py",
            file=sys.stderr,
        )
        return 1

    print(
        f"=== Starting API :{backend_port} + UX :{ux_port} (CDSW_APP_PORT) ==="
    )
    ensure_api_running(backend_port)
    if not wait_for_urls([api_health_url(backend_port)], attempts=60):
        print(f"API not ready on :{backend_port}; see {log_file('api')}", file=sys.stderr)
        return 1

    if is_cai_session():
        ip = cai_session_ip()
        print(f"CAI session IP: {ip}")
        print(f"Open the Application URL → UX on CDSW_APP_PORT={ux_port}")
        print(f"API (internal): http://127.0.0.1:{backend_port}")
        print(
            f"Smoke test: API_BASE=http://127.0.0.1:{backend_port} "
            f"{venv_python()} {REPO_ROOT / 'scripts' / 'cai_smoke_test.py'}"
        )
        return run_ui_gateway_foreground(ux_port, backend_port)

    start_ui_gateway_background(ux_port, backend_port)
    if wait_for_urls(
        [api_health_url(backend_port), f"http://127.0.0.1:{ux_port}/"],
        attempts=60,
    ):
        print(f"UX: http://127.0.0.1:{ux_port}/  API: http://127.0.0.1:{backend_port}/docs")
        return 0
    print(f"Timed out; see {log_file('api')} and {log_file('ui')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
