"""Versioned task configurations (append-only specs)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Experiment, TaskDefinition, TaskVersion


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def content_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()


def list_tasks_for_project(db: Session, project_id: int) -> list[TaskDefinition]:
    return list(
        db.scalars(
            select(TaskDefinition)
            .where(TaskDefinition.project_id == project_id)
            .order_by(TaskDefinition.created_at.desc())
        )
    )


def get_task(db: Session, task_id: int) -> TaskDefinition | None:
    return db.get(TaskDefinition, task_id)


def create_task(db: Session, *, project_id: int, name: str) -> TaskDefinition:
    project = db.get(Experiment, project_id)
    if not project:
        raise ValueError("Project not found")
    task = TaskDefinition(project_id=project_id, name=name.strip())
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def list_versions(db: Session, task_id: int) -> list[TaskVersion]:
    return list(
        db.scalars(
            select(TaskVersion)
            .where(TaskVersion.task_id == task_id)
            .order_by(TaskVersion.version.desc())
        )
    )


def get_version(db: Session, version_id: int) -> TaskVersion | None:
    return db.get(TaskVersion, version_id)


def create_version(
    db: Session,
    *,
    task_id: int,
    spec: dict[str, Any],
) -> TaskVersion:
    task = db.get(TaskDefinition, task_id)
    if not task:
        raise ValueError("Task not found")
    h = content_hash(spec)
    latest = db.scalar(
        select(func.max(TaskVersion.version)).where(TaskVersion.task_id == task_id)
    )
    next_v = int(latest or 0) + 1
    row = TaskVersion(
        task_id=task_id,
        version=next_v,
        content_hash=h,
        spec_json=json.dumps(spec, indent=2),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def create_task_with_version(
    db: Session,
    *,
    project_id: int,
    task_name: str,
    spec: dict[str, Any],
) -> tuple[TaskDefinition, TaskVersion]:
    task = create_task(db, project_id=project_id, name=task_name)
    version = create_version(db, task_id=task.id, spec=spec)
    return task, version
