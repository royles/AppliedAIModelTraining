#!/usr/bin/env python3
"""Compile frontend/src → frontend/dist (production bundle served by ui_gateway)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, cd_repo, frontend_dist_ready, run, write_ui_build_stamp
from install import build_frontend


def main() -> int:
    cd_repo()
    if not shutil.which("npm"):
        print("npm not found; install Node.js or copy a prebuilt frontend/dist.", file=sys.stderr)
        return 1
    if not build_frontend(force=True):
        return 1
    if frontend_dist_ready():
        write_ui_build_stamp()
        print(f"Production UI ready at {REPO_ROOT / 'frontend' / 'dist'}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
