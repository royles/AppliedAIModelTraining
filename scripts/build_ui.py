#!/usr/bin/env python3
"""
Compile frontend/src → frontend/dist.

On CAI/CDSW (no Node/npm), this script exits successfully when the repo already
contains a current prebuilt frontend/dist — use `git pull` instead of compiling.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    REPO_ROOT,
    cd_repo,
    ensure_prebuilt_ui_or_npm,
    frontend_dist_ready,
    ui_bundle_has_sidebar_ux,
    write_ui_build_stamp,
)
from install import build_frontend


def main() -> int:
    cd_repo()

    if not shutil.which("npm"):
        ok, message = ensure_prebuilt_ui_or_npm()
        print(message)
        if frontend_dist_stale():
            print(
                "\nCannot rebuild UI on this machine (npm not installed).",
                file=sys.stderr,
            )
            print(
                "  On CAI/CDSW: git pull origin main  — production UI lives in frontend/dist.",
                file=sys.stderr,
            )
            print(
                "  Off CAI: run npm run build in frontend/, commit dist, or use a machine with Node.",
                file=sys.stderr,
            )
        return 0 if ok else 1

    if frontend_dist_ready() and ui_bundle_has_sidebar_ux():
        import sys as _sys

        if "--force" not in _sys.argv[1:]:
            print(
                "Production UI already built in frontend/dist. "
                "Pass --force to run npm run build again."
            )
            return 0

    if not build_frontend(force=True):
        return 1
    if frontend_dist_ready():
        write_ui_build_stamp()
        print(f"Production UI ready at {REPO_ROOT / 'frontend' / 'dist'}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
