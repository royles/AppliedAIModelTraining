"""Per-project Ray + CDSW worker configuration (one cluster target per project)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Experiment, PlatformSettings, ProjectRaySettings
from app.schemas import (
    CdswWorkerSettings,
    ProjectRaySettingsRead,
    ProjectRaySettingsUpdate,
    RaySettings,
)
from app.services.platform_settings import (
    _cdsw_settings,
    _ray_settings,
    ensure_platform_settings,
    runtime_info,
)


def _copy_platform_to_project_row(platform: PlatformSettings) -> dict[str, Any]:
    return {
        "ray_local_mode": platform.ray_local_mode,
        "ray_head_address": platform.ray_head_address,
        "ray_train_num_workers": platform.ray_train_num_workers,
        "non_distributed_low_memory_mode": platform.non_distributed_low_memory_mode,
        "worker_cpu": platform.worker_cpu,
        "worker_memory_gb": platform.worker_memory_gb,
        "gpus_per_worker": platform.gpus_per_worker,
        "default_job_num_workers": platform.default_job_num_workers,
        "accelerator_label_id": platform.accelerator_label_id,
        "resource_profile_id": platform.resource_profile_id,
        "resource_group_id": platform.resource_group_id,
    }


def seed_project_ray_settings(db: Session, project_id: int) -> ProjectRaySettings:
    existing = db.get(ProjectRaySettings, project_id)
    if existing is not None:
        return existing
    if not db.get(Experiment, project_id):
        raise ValueError("Project not found")
    platform = ensure_platform_settings(db)
    row = ProjectRaySettings(project_id=project_id, **_copy_platform_to_project_row(platform))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def seed_missing_project_ray_settings(db: Session) -> int:
    project_ids = list(db.scalars(select(Experiment.id)))
    created = 0
    for pid in project_ids:
        if db.get(ProjectRaySettings, pid) is None:
            seed_project_ray_settings(db, pid)
            created += 1
    return created


def ensure_project_ray_settings(db: Session, project_id: int) -> ProjectRaySettings:
    row = db.get(ProjectRaySettings, project_id)
    if row is not None:
        return row
    return seed_project_ray_settings(db, project_id)


def get_project_ray_settings(db: Session, project_id: int) -> ProjectRaySettingsRead:
    if not db.get(Experiment, project_id):
        raise ValueError("Project not found")
    row = ensure_project_ray_settings(db, project_id)
    return ProjectRaySettingsRead(
        project_id=project_id,
        ray=_ray_settings(row),
        cdsw_workers=_cdsw_settings(row),
        runtime=runtime_info(),
    )


def update_project_ray_settings(
    db: Session, project_id: int, payload: ProjectRaySettingsUpdate
) -> ProjectRaySettingsRead:
    if not db.get(Experiment, project_id):
        raise ValueError("Project not found")
    row = ensure_project_ray_settings(db, project_id)
    if payload.ray is not None:
        data = payload.ray.model_dump()
        row.ray_local_mode = data["ray_local_mode"]
        row.ray_head_address = data["ray_head_address"]
        row.ray_train_num_workers = data["ray_train_num_workers"]
        row.non_distributed_low_memory_mode = data["non_distributed_low_memory_mode"]
    if payload.cdsw_workers is not None:
        data = payload.cdsw_workers.model_dump()
        for key, value in data.items():
            setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return get_project_ray_settings(db, project_id)


def delete_project_ray_settings(db: Session, project_id: int) -> None:
    row = db.get(ProjectRaySettings, project_id)
    if row is not None:
        db.delete(row)
