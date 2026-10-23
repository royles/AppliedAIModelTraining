"""CAI Application / ipykernel argv handling for scripts/start.py and install.py."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _import_paths():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from _paths import strip_jupyter_kernel_argv

    return strip_jupyter_kernel_argv


def test_strip_jupyter_connection_file_pair():
    strip = _import_paths()
    argv = ["--install", "-f", "/tmp/jupyter/runtime/kernel-deadbeef.json"]
    assert strip(argv) == ["--install"]


def test_install_parse_ignores_start_flags():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import install

    saved = list(sys.argv)
    try:
        sys.argv = [
            "scripts/start.py",
            "-f",
            "/tmp/jupyter/runtime/kernel-x.json",
            "--install",
            "--rebuild-ui",
        ]
        args = install.parse_args()
        assert args.rebuild_ui is True
    finally:
        sys.argv = saved
