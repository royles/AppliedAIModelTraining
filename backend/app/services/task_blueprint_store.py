"""Persist model-task blueprint JSON on disk (draft + named blueprints)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT

BLUEPRINTS_DIR = (REPO_ROOT / "data" / "blueprints").resolve()
DRAFT_FILENAME = "_draft.json"
SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir() -> None:
    BLUEPRINTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-").lower()
    return slug[:120] or "blueprint"


def draft_path() -> Path:
    return BLUEPRINTS_DIR / DRAFT_FILENAME


def read_draft() -> dict[str, Any] | None:
    path = draft_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_draft(body: dict[str, Any]) -> dict[str, Any]:
    _ensure_dir()
    payload = dict(body)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("kind", "model_task_blueprint")
    payload["updated_at"] = _now_iso()
    path = draft_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return payload


def list_named_blueprints() -> list[dict[str, Any]]:
    _ensure_dir()
    out: list[dict[str, Any]] = []
    for path in sorted(BLUEPRINTS_DIR.glob("*.json")):
        if path.name == DRAFT_FILENAME:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        out.append(
            {
                "id": path.stem,
                "name": data.get("name") or path.stem,
                "updated_at": data.get("updated_at"),
                "preset": data.get("preset"),
            }
        )
    out.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
    return out


def load_named(blueprint_id: str) -> dict[str, Any]:
    _ensure_dir()
    clean = _safe_id(blueprint_id)
    path = (BLUEPRINTS_DIR / f"{clean}.json").resolve()
    if not str(path).startswith(str(BLUEPRINTS_DIR)) or not path.is_file():
        raise FileNotFoundError(f"Blueprint not found: {blueprint_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid blueprint file")
    return data


def save_named(name: str, body: dict[str, Any]) -> dict[str, Any]:
    _ensure_dir()
    blueprint_id = _safe_id(name)
    payload = dict(body)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("kind", "model_task_blueprint")
    payload["name"] = name.strip() or blueprint_id
    payload["updated_at"] = _now_iso()
    path = BLUEPRINTS_DIR / f"{blueprint_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"id": blueprint_id, "name": payload["name"], "updated_at": payload["updated_at"]}


def delete_named(blueprint_id: str) -> None:
    path = (BLUEPRINTS_DIR / f"{_safe_id(blueprint_id)}.json").resolve()
    if not str(path).startswith(str(BLUEPRINTS_DIR)):
        raise ValueError("Invalid blueprint id")
    path.unlink(missing_ok=True)


def save_job_blueprint(job_id: int, body: dict[str, Any]) -> Path:
    _ensure_dir()
    jobs_dir = (REPO_ROOT / "data" / "models" / f"job_{job_id}").resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(body)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("kind", "model_task_blueprint")
    payload["job_id"] = job_id
    payload["updated_at"] = _now_iso()
    path = jobs_dir / "task_blueprint.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
