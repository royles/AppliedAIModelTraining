#!/usr/bin/env python3
"""
Start the platform (split ports, 127.0.0.1 only):

- **UX** — single bind on CDSW_APP_PORT (CAI) or PLATFORM_UI_PORT (local)
- **API** — PLATFORM_API_PORT (default 8000)

CDSW_APP_PORT is read once from the environment and only used for the UX gateway.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BIND_HOST,
    REPO_ROOT,
    api_health_url,
    api_port,
    cai_session_ip,
    cd_repo,
    ensure_api_running,
    ensure_runtime_ready,
    frontend_dist_ready,
    ensure_bundled_datasets,
    ensure_prebuilt_ui_or_npm,
    frontend_dist_stale,
    read_ui_build_label,
    is_cai_session,
    log_file,
    print_cai_memory_hints,
    resolve_ux_port,
    run_ui_gateway_foreground,
    run_vite_dev_foreground,
    start_ray_head_background,
    start_ui_gateway_background,
    ui_api_health_url,
    venv_python,
    wait_for_urls,
)
from install import build_frontend, main as install_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Applied AI Model Training services")
    parser.add_argument("--install", action="store_true", help="Run install.py first")
    parser.add_argument(
        "--dev-ui",
        action="store_true",
        help="Vite dev server on UX port (npm) + API on PLATFORM_API_PORT",
    )
    parser.add_argument(
        "--rebuild-ui",
        action="store_true",
        help="Run npm run build into frontend/dist (also with --install)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cd_repo()

    try:
        ux_port = resolve_ux_port()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    backend_port = api_port()

    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)
    ensure_bundled_datasets()

    code = ensure_runtime_ready()
    if code != 0:
        return code

    if args.rebuild_ui:
        if shutil.which("npm"):
            if build_frontend(force=True) is False:
                return 1
        else:
            ok, message = ensure_prebuilt_ui_or_npm()
            print(message)
            if not ok:
                return 1

    if args.install:
        if args.rebuild_ui:
            import os as _os

            _os.environ["PLATFORM_UI_REBUILD"] = "1"
        code = install_main()
        if code != 0:
            return code
        code = ensure_runtime_ready()
        if code != 0:
            return code

    if is_cai_session():
        os.environ["RAY_LOCAL_MODE"] = "false"
        print_cai_memory_hints()

    if args.dev_ui:
        print(f"=== Dev UI: {BIND_HOST}:{ux_port}, API {BIND_HOST}:{backend_port} ===")
        ensure_api_running(backend_port)
        if not wait_for_urls([api_health_url(backend_port)], attempts=60):
            print(f"API not ready; see {log_file('api')}", file=sys.stderr)
            return 1
        try:
            return run_vite_dev_foreground(ux_port)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    ok, ui_message = ensure_prebuilt_ui_or_npm()
    if not ok:
        print(ui_message, file=sys.stderr)
        return 1
    if not args.dev_ui:
        print(ui_message)

    if not args.dev_ui and frontend_dist_stale():
        if shutil.which("npm"):
            fix = "python3 scripts/build_ui.py --force   or   python3 scripts/start.py --rebuild-ui"
        else:
            fix = "git pull origin main  (build_ui.py cannot compile without npm on this machine)"
        print(
            "WARNING: frontend/dist does not match frontend/src (see .ui-build-stamp).\n"
            f"  Fix: {fix}",
            file=sys.stderr,
        )
    elif not args.dev_ui:
        label = read_ui_build_label()
        if label:
            print(f"Production UI bundle: {label}")

    print(
        f"=== API {BIND_HOST}:{backend_port} + UX {BIND_HOST}:{ux_port} "
        f"(CDSW_APP_PORT={ux_port if is_cai_session() else 'n/a'}) ==="
    )
    ensure_api_running(backend_port)
    if not wait_for_urls([api_health_url(backend_port)], attempts=60):
        print(
            f"API not ready on :{backend_port}; see {log_file('api')}",
            file=sys.stderr,
        )
        print(
            f"Tip: {venv_python()} -m pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        return 1

    start_ui_gateway_background(ux_port, backend_port)
    ux_ready = wait_for_urls([ui_api_health_url(ux_port)], attempts=60)
    if ux_ready:
        print(
            f"UX ready: http://{BIND_HOST}:{ux_port}/  "
            f"(API via /api → {BIND_HOST}:{backend_port})"
        )
    else:
        print(
            f"UX not responding on :{ux_port}; see {log_file('ui')}",
            file=sys.stderr,
        )

    if is_cai_session():
        start_ray_head_background()
        ip = cai_session_ip()
        print(f"CAI session IP: {ip}")
        print(f"Application route → UX on CDSW_APP_PORT={ux_port} ({BIND_HOST})")
        print(f"Smoke test: API_BASE=http://{BIND_HOST}:{backend_port} "
              f"{venv_python()} {REPO_ROOT / 'scripts' / 'cai_smoke_test.py'}")
        if ensure_runtime_ready() != 0:
            return 1
        if not ux_ready:
            return 1
        return run_ui_gateway_foreground(ux_port, backend_port)

    if ux_ready:
        print(f"Direct API docs: http://{BIND_HOST}:{backend_port}/docs")
        return 0
    print(f"Timed out; see {log_file('api')} and {log_file('ui')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
