"""Locate ``scripts/`` when the entrypoint is a file, ``python scripts/start.py``, or a CAI notebook cell."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_scripts_on_sys_path() -> Path:
    """
    Add the directory containing ``common.py`` to ``sys.path``.

    In Cloudera AI Applications / Jupyter, ``__file__`` is undefined in notebook
    cells — use ``CDSW_PROJECT_ROOT`` or discover ``scripts/common.py`` from cwd.
    """
    for scripts in _scripts_dir_candidates():
        if (scripts / "common.py").is_file():
            entry = str(scripts)
            if entry not in sys.path:
                sys.path.insert(0, entry)
            return scripts
    raise RuntimeError(
        "Could not find scripts/common.py. Set CDSW_PROJECT_ROOT to the repo root, "
        "or run from the project directory (Applications: use python3 scripts/start.py, "
        "not a pasted line with __file__)."
    )


def _scripts_dir_candidates() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p.resolve())

    try:
        add(Path(__file__).resolve().parent)
    except NameError:
        pass

    for env_key in ("CDSW_PROJECT_ROOT", "PROJECT_ROOT"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            add(Path(raw) / "scripts")
            add(Path(raw))

    cwd = Path.cwd().resolve()
    add(cwd / "scripts")
    add(cwd)
    if cwd.name == "scripts" and (cwd / "common.py").is_file():
        add(cwd)

    return out
