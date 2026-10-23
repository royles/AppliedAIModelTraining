#!/usr/bin/env python3
"""Install Python dependencies and build the React UI into frontend/dist."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import os
import sys
from pathlib import Path


def _ensure_scripts_on_syspath() -> None:
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    for key in ("CDSW_PROJECT_ROOT", "PROJECT_ROOT"):
        root = os.environ.get(key, "").strip()
        if root:
            candidates.append(Path(root) / "scripts")
            candidates.append(Path(root))
    candidates.append(Path.cwd() / "scripts")
    candidates.append(Path.cwd())
    for scripts in candidates:
        scripts = scripts.resolve()
        if (scripts / "common.py").is_file():
            entry = str(scripts)
            if entry not in sys.path:
                sys.path.insert(0, entry)
            return
    raise RuntimeError("Could not find scripts/common.py; set CDSW_PROJECT_ROOT or run from repo root.")


_ensure_scripts_on_syspath()

from common import (
    REPO_ROOT,
    cd_repo,
    ensure_bundled_datasets,
    frontend_dist_ready,
    frontend_dist_stale,
    pip_install,
    platform_python,
    run,
    write_ui_build_stamp,
)


def _ui_rebuild_requested(force: bool) -> bool:
    if force:
        return True
    env = os.environ.get("PLATFORM_UI_REBUILD", "").strip().lower()
    return env in ("1", "true", "yes", "force")


def build_frontend(force: bool = False) -> bool:
    if (
        frontend_dist_ready()
        and not _ui_rebuild_requested(force)
        and not frontend_dist_stale()
    ):
        print("frontend/dist is up to date; skipping npm build.")
        return True

    if frontend_dist_ready() and not _ui_rebuild_requested(force) and frontend_dist_stale():
        print("frontend/dist is older than frontend/src — rebuilding production UI…")
    elif _ui_rebuild_requested(force):
        print("Rebuilding production UI (forced)…")

    if not shutil.which("npm"):
        if frontend_dist_ready():
            from common import ui_bundle_has_sidebar_ux

            if frontend_dist_stale() and not ui_bundle_has_sidebar_ux():
                print(
                    "ERROR: npm not found and frontend/dist is missing or outdated.\n"
                    "  On CAI: git pull origin main  (production UI is committed in frontend/dist)\n"
                    "  Off CAI: npm run build in frontend/ and commit dist, or install Node here.",
                    file=sys.stderr,
                )
                return False
            print(
                "npm not found; using prebuilt frontend/dist from the repo (no compile step).",
            )
            return True
        print(
            "npm not found: cannot build the UI. Use a machine with Node to run "
            "'python3 scripts/build_ui.py' or 'npm run build' in frontend/.",
            file=sys.stderr,
        )
        return False

    run(["npm", "install"], cwd=REPO_ROOT / "frontend")
    run(["npm", "run", "build"], cwd=REPO_ROOT / "frontend")
    if not frontend_dist_ready():
        print("npm run build did not produce frontend/dist/index.html", file=sys.stderr)
        return False
    write_ui_build_stamp()
    print("Built React app to frontend/dist (served by FastAPI; no npm at start time).")
    return True


def install_python_deps() -> None:
    """Install backend requirements into the active system Python."""
    print(f"Installing Python deps: {platform_python()}")
    try:
        pip_install(["--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print(
            "Warning: could not upgrade pip (common on CDSW); continuing.",
            file=sys.stderr,
        )
    pip_install(["-r", "backend/requirements.txt"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install platform dependencies")
    parser.add_argument(
        "--rebuild-ui",
        action="store_true",
        help="Always run npm run build (refresh frontend/dist)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cd_repo()
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)
    ensure_bundled_datasets()

    install_python_deps()

    if not build_frontend(force=args.rebuild_ui):
        if frontend_dist_ready():
            print("Install complete (using existing frontend/dist).")
            return 0
        return 1

    print("Install complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
