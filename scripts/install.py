#!/usr/bin/env python3
"""Install Python and Node dependencies for the platform."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, cd_repo, ensure_python_venv, pip_install, run


def main() -> int:
    cd_repo()
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "backend" / "data").mkdir(parents=True, exist_ok=True)

    ensure_python_venv()

    try:
        pip_install(["--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print(
            "Warning: could not upgrade pip (common on CDSW); continuing.",
            file=sys.stderr,
        )

    pip_install(["-r", "backend/requirements.txt"])

    if not shutil.which("npm"):
        print("npm is required for the React UI.", file=sys.stderr)
        return 1

    run(["npm", "install"], cwd=REPO_ROOT / "frontend")
    print("Install complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
