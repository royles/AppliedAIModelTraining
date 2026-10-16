from __future__ import annotations

import re
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TrainedModel

OPENAI_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def validate_openai_model_id(model_id: str) -> str:
    if not OPENAI_ID_PATTERN.match(model_id):
        raise ValueError(
            "openai_model_id must be 1-64 chars: letters, digits, dots, dashes, underscores"
        )
    return model_id


def default_openai_model_id(job_id: int, name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")[:40] or "model"
    return validate_openai_model_id(f"{slug}-{job_id}")


def register_trained_model(
    db: Session,
    *,
    job_id: int,
    openai_model_id: str,
    artifact_path: str,
    base_model: str,
    task_type: str,
    architecture_json: str | None = None,
) -> TrainedModel:
    openai_model_id = validate_openai_model_id(openai_model_id)
    path = Path(artifact_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact path not found: {artifact_path}")

    existing = db.scalar(
        select(TrainedModel).where(TrainedModel.openai_model_id == openai_model_id)
    )
    if existing:
        raise ValueError(f"openai_model_id already registered: {openai_model_id}")

    record = TrainedModel(
        job_id=job_id,
        openai_model_id=openai_model_id,
        artifact_path=str(path),
        base_model=base_model,
        task_type=task_type,
        architecture_json=architecture_json,
        created_at_unix=int(time.time()),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_model_by_openai_id(db: Session, model_id: str) -> TrainedModel | None:
    return db.scalar(
        select(TrainedModel).where(TrainedModel.openai_model_id == model_id)
    )


def list_models(db: Session) -> list[TrainedModel]:
    return list(
        db.scalars(select(TrainedModel).order_by(TrainedModel.created_at_unix.desc()))
    )


def get_trained_model_for_job(db: Session, job_id: int) -> TrainedModel | None:
    return db.scalar(select(TrainedModel).where(TrainedModel.job_id == job_id))


def openai_model_ids_for_jobs(db: Session, job_ids: list[int]) -> dict[int, str]:
    if not job_ids:
        return {}
    rows = db.execute(
        select(TrainedModel.job_id, TrainedModel.openai_model_id).where(
            TrainedModel.job_id.in_(job_ids)
        )
    )
    return {int(job_id): model_id for job_id, model_id in rows.all()}
