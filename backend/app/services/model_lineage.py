"""Immutable lineage snapshots for registry entries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.models import Experiment, TaskDefinition, TaskVersion, TrainingJob
from app.services.platform_settings import _cdsw_settings, _ray_settings
from app.services.project_ray_settings import ensure_project_ray_settings
from app.services.worker_resources import _worker_resources_from_config, resolve_worker_launch_spec


def load_job_blueprint_file(job_id: int) -> dict[str, Any] | None:
    path = REPO_ROOT / "data" / "models" / f"job_{job_id}" / "task_blueprint.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _ray_config_snapshot(db: Session, project_id: int) -> dict[str, Any]:
    row = ensure_project_ray_settings(db, project_id)
    return {
        "project_id": project_id,
        "ray": _ray_settings(row).model_dump(),
        "cdsw_workers": _cdsw_settings(row).model_dump(),
    }


def _job_run_snapshot(db: Session, job: TrainingJob) -> dict[str, Any]:
    ray_row = ensure_project_ray_settings(db, job.experiment_id)
    worker_spec = resolve_worker_launch_spec(job, ray_row)
    wg = job.worker_group
    cdsw_worker_ids: list[str] | None = None
    if wg and wg.worker_ids_json:
        try:
            parsed = json.loads(wg.worker_ids_json)
            if isinstance(parsed, list):
                cdsw_worker_ids = [str(x) for x in parsed]
        except json.JSONDecodeError:
            cdsw_worker_ids = None

    worker_group: dict[str, Any] | None = None
    if wg is not None:
        worker_group = {
            "status": wg.status,
            "target_workers": wg.target_workers,
            "head_address": wg.head_address,
            "cdsw_worker_ids": cdsw_worker_ids,
        }

    overrides = _worker_resources_from_config(job.config_json)

    return {
        "status": job.status,
        "ray_address": job.ray_address,
        "num_workers": job.num_workers,
        "gpus_per_worker": job.gpus_per_worker,
        "epochs": job.epochs,
        "batch_size": job.batch_size,
        "learning_rate": job.learning_rate,
        "max_steps": job.max_steps,
        "worker_resources_override": overrides or None,
        "effective_worker_launch_spec": worker_spec,
        "worker_group": worker_group,
        "completed_at": job.updated_at.isoformat() if job.updated_at else None,
    }


def build_lineage_snapshot(db: Session, job: TrainingJob) -> dict[str, Any]:
    job = db.get(TrainingJob, job.id)
    if not job:
        raise ValueError("Job not found")
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
        "ray_config": _ray_config_snapshot(db, job.experiment_id),
        "job_run": _job_run_snapshot(db, job),
    }


def card_content_hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()
