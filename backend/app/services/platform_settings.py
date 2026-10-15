"""Persisted Ray / CDSW worker defaults (SQLite), with env fallbacks."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import PlatformSettings
from app.schemas import (
    CdswWorkerSettings,
    PlatformRuntimeInfo,
    PlatformSettingsRead,
    PlatformSettingsUpdate,
    RaySettings,
)
from app.services import cdsw_workers


def _env_defaults() -> dict[str, Any]:
    settings = get_settings()
    return {
        "ray_local_mode": settings.ray_local_mode,
        "ray_head_address": settings.cdsw_head_address or os.environ.get("RAY_HEAD_ADDRESS"),
        "ray_train_num_workers": int(os.environ.get("RAY_TRAIN_NUM_WORKERS", "0") or 0),
        "worker_cpu": settings.default_worker_cpu,
        "worker_memory_gb": settings.default_worker_memory_gb,
        "gpus_per_worker": settings.default_gpus_per_worker,
        "default_job_num_workers": 1,
        "accelerator_label_id": os.environ.get("CML_ACCELERATOR_LABEL_ID"),
        "resource_profile_id": os.environ.get("CML_RESOURCE_PROFILE_ID"),
        "resource_group_id": os.environ.get("CML_RESOURCE_GROUP_ID"),
    }


def ensure_platform_settings(db: Session) -> PlatformSettings:
    row = db.get(PlatformSettings, 1)
    if row is not None:
        return row
    defaults = _env_defaults()
    row = PlatformSettings(id=1, **defaults)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _ray_settings(row: PlatformSettings) -> RaySettings:
    return RaySettings(
        ray_local_mode=row.ray_local_mode,
        ray_head_address=row.ray_head_address,
        ray_train_num_workers=row.ray_train_num_workers,
    )


def _cdsw_settings(row: PlatformSettings) -> CdswWorkerSettings:
    return CdswWorkerSettings(
        worker_cpu=row.worker_cpu,
        worker_memory_gb=row.worker_memory_gb,
        gpus_per_worker=row.gpus_per_worker,
        default_job_num_workers=row.default_job_num_workers,
        accelerator_label_id=row.accelerator_label_id,
        resource_profile_id=row.resource_profile_id,
        resource_group_id=row.resource_group_id,
    )


def runtime_info() -> PlatformRuntimeInfo:
    import ray

    runtime = "initialized" if ray.is_initialized() else "not_initialized"
    head = cdsw_workers.ray_head_address()
    return PlatformRuntimeInfo(
        ray_runtime=runtime,
        cdsw_available=cdsw_workers.cdsw_available(),
        detected_ray_head=head,
    )


def get_platform_settings(db: Session) -> PlatformSettingsRead:
    row = ensure_platform_settings(db)
    return PlatformSettingsRead(
        ray=_ray_settings(row),
        cdsw_workers=_cdsw_settings(row),
        runtime=runtime_info(),
    )


def update_platform_settings(
    db: Session, payload: PlatformSettingsUpdate
) -> PlatformSettingsRead:
    row = ensure_platform_settings(db)
    if payload.ray is not None:
        data = payload.ray.model_dump()
        row.ray_local_mode = data["ray_local_mode"]
        row.ray_head_address = data["ray_head_address"]
        row.ray_train_num_workers = data["ray_train_num_workers"]
    if payload.cdsw_workers is not None:
        data = payload.cdsw_workers.model_dump()
        for key, value in data.items():
            setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return get_platform_settings(db)


def resolved_ray_head(row: PlatformSettings | None) -> str | None:
    if row and row.ray_head_address:
        return row.ray_head_address
    return cdsw_workers.ray_head_address() or get_settings().cdsw_head_address


def worker_pool_kwargs(row: PlatformSettings) -> dict[str, Any]:
    return {
        "cpu": row.worker_cpu,
        "memory_gb": row.worker_memory_gb,
        "gpus_per_worker": row.gpus_per_worker,
        "accelerator_label_id": row.accelerator_label_id,
        "resource_profile_id": row.resource_profile_id,
        "resource_group_id": row.resource_group_id,
    }
