"""Read platform release version from repo VERSION file (x.y.z.b)."""

from __future__ import annotations

import re

from app.config import REPO_ROOT

_VERSION_LINE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s*$")
_DEFAULT = "1.1.0.0"


def read_platform_version() -> str:
    path = REPO_ROOT / "VERSION"
    if not path.is_file():
        return _DEFAULT
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if _VERSION_LINE.match(line):
            return line
    return _DEFAULT
