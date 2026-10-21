"""Immutable lineage snapshots for registry entries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.models import Experiment, TaskDefinition, TaskVersion, TrainingJob


def load_job_blueprint_file(job_id: int) -> dict[str, Any] | None:
    path = REPO_ROOT / "data" / "models" / f"job_{job_id}" / "task_blueprint.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def build_lineage_snapshot(db: Session, job: TrainingJob) -> dict[str, Any]:
    project = db.get(Experiment, job.experiment_id)
    blueprint = load_job_blueprint_file(job.id)
    task_id: int | None = None
    task_name: str | None = None
    task_version: int | None = None
    content_hash: str | None = None
    task_version_id: int | None = job.task_version_id

    if job.task_version_id:
        ver = db.get(TaskVersion, job.task_version_id)
        if ver:
            task_version_id = ver.id
            task_version = ver.version
            content_hash = ver.content_hash
            task = db.get(TaskDefinition, ver.task_id)
            if task:
                task_id = task.id
                task_name = task.name
            if blueprint is None:
                try:
                    blueprint = json.loads(ver.spec_json)
                except json.JSONDecodeError:
                    pass

    return {
        "project_id": job.experiment_id,
        "project_name": project.name if project else None,
        "task_id": task_id,
        "task_name": task_name,
        "task_version_id": task_version_id,
        "task_version": task_version,
        "content_hash": content_hash,
        "job_id": job.id,
        "job_name": job.name,
        "job_created_at": job.created_at.isoformat() if job.created_at else None,
        "blueprint": blueprint,
    }


def card_content_hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()
