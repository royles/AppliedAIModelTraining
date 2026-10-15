#!/usr/bin/env python3
"""Install Python dependencies and build the React UI into frontend/dist."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, cd_repo, frontend_dist_ready, pip_install, platform_python, run


def build_frontend() -> bool:
    if frontend_dist_ready():
        print("frontend/dist already present; skipping npm build.")
        return True
    if not shutil.which("npm"):
        print(
            "npm not found: cannot build the UI. Use a machine with Node to run "
            "'npm install && npm run build' in frontend/, or copy frontend/dist into the project.",
            file=sys.stderr,
        )
        return False
    run(["npm", "install"], cwd=REPO_ROOT / "frontend")
    run(["npm", "run", "build"], cwd=REPO_ROOT / "frontend")
    if not frontend_dist_ready():
        print("npm run build did not produce frontend/dist/index.html", file=sys.stderr)
        return False
    print("Built React app to frontend/dist (served by FastAPI; no npm at start time).")
    return True


def main() -> int:
    cd_repo()
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)

    print(f"Installing into system Python: {platform_python()}")

    try:
        pip_install(["--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print(
            "Warning: could not upgrade pip (common on CDSW); continuing.",
            file=sys.stderr,
        )

    pip_install(["-r", "backend/requirements.txt"])

    if not build_frontend():
        if frontend_dist_ready():
            print("Install complete (using existing frontend/dist).")
            return 0
        return 1

    print("Install complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
