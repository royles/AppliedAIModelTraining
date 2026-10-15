#!/usr/bin/env python3
"""Bootstrap Ray + platform services in a Cloudera AI workbench session."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, cd_repo, run, venv_python


def main() -> int:
    cd_repo()
    os.environ["RAY_LOCAL_MODE"] = "false"

    ip = os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")
    if ip:
        if not os.environ.get("CDSW_APP_PORT"):
            print(
                "CDSW_APP_PORT must be set in a Cloudera AI Application / workbench session.",
                file=sys.stderr,
            )
            return 1
        os.environ["RAY_HEAD_ADDRESS"] = f"{ip}:6379"
        print(f"CAI session detected at {ip}; starting Ray head...")
        run(
            [
                "ray",
                "start",
                "--head",
                "--port=6379",
                "--dashboard-port=8265",
                "--include-dashboard=true",
            ],
            check=False,
        )

    py = venv_python()
    scripts = REPO_ROOT / "scripts"
    run([py, str(scripts / "install.py")])
    run([py, str(scripts / "start.py")])

    port = os.environ.get("CDSW_APP_PORT", "")
    print("API: http://127.0.0.1:8000/docs")
    print(f"UI:  bound to CDSW_APP_PORT={port} (use your CAI Application route URL)")
    print(f"Run: {py} {scripts / 'cai_smoke_test.py'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
