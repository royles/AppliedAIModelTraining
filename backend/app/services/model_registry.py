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
