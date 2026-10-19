"""On-disk training progress for job restart / resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def progress_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "progress.json"


def read_progress(artifact_dir: str | Path) -> dict[str, Any] | None:
    path = progress_path(artifact_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_progress(artifact_dir: str | Path, payload: dict[str, Any]) -> None:
    path = progress_path(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def latest_checkpoint_dir(artifact_dir: str | Path) -> Path | None:
    root = Path(artifact_dir) / "checkpoints"
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("epoch_")]
    if not candidates:
        return None

    def _epoch_num(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=lambda p: _epoch_num(p.name))


def apply_resume_to_train_config(
    train_config: dict[str, Any],
    artifact_dir: str | Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Merge resume metadata into train_config when checkpoints exist."""
    if not resume:
        train_config.pop("resume", None)
        train_config.pop("resume_global_step", None)
        train_config.pop("resume_checkpoint_path", None)
        return train_config

    progress = read_progress(artifact_dir)
    ckpt = latest_checkpoint_dir(artifact_dir)
    if not progress and not ckpt:
        return {**train_config, "resume": False}

    out = {**train_config, "resume": True}
    if progress:
        done = int(progress.get("global_step") or 0)
        target = int(progress.get("max_steps") or train_config.get("max_steps") or 0)
        out["resume_global_step"] = done
        out["resume_epoch"] = int(progress.get("epoch") or 0)
        if target > done:
            out["max_steps"] = target
        out["remaining_steps"] = max(0, int(out.get("max_steps", target)) - done)
    if ckpt is not None:
        out["resume_checkpoint_path"] = str(ckpt)
    tok = Path(artifact_dir) / "tokenizer"
    if tok.is_dir():
        out["tokenizer_path"] = str(tok)
    return out
