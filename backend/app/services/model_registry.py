from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TrainedModel, TrainingJob
from app.services.model_card import build_training_metadata, generate_model_card
from app.services.model_lifecycle import (
    STATE_BENCHMARKED,
    STATE_RETIRED,
    assert_transition,
    state_guidance,
)

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
    job: TrainingJob,
    openai_model_id: str,
    artifact_path: str,
    base_model: str,
    task_type: str,
    architecture_json: str | None = None,
    init_mode: str | None = None,
    training_metrics: dict[str, Any] | None = None,
    parent_model_id: int | None = None,
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

    metadata = build_training_metadata(
        job=job,
        config_json=job.config_json,
        metrics=training_metrics,
        artifact_path=str(path),
        init_mode=init_mode,
    )
    now = datetime.now(timezone.utc)
    record = TrainedModel(
        job_id=job.id,
        openai_model_id=openai_model_id,
        artifact_path=str(path),
        base_model=base_model,
        task_type=task_type,
        architecture_json=architecture_json,
        created_at_unix=int(time.time()),
        lifecycle_state=STATE_TRAINED,
        display_name=job.name,
        metadata_json=json.dumps(metadata),
        parent_model_id=parent_model_id,
        updated_at=now,
    )
    db.add(record)
    db.flush()
    record.model_card_markdown = generate_model_card(record, job=job)
    db.commit()
    db.refresh(record)
    return record


def transition_model_state(db: Session, registry_id: int, new_state: str) -> TrainedModel:
    record = db.get(TrainedModel, registry_id)
    if not record:
        raise ValueError("Registry entry not found")
    assert_transition(record.lifecycle_state, new_state)
    record.lifecycle_state = new_state
    record.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return record


def update_registry_entry(
    db: Session,
    registry_id: int,
    *,
    display_name: str | None = None,
    description: str | None = None,
    benchmark: dict[str, Any] | None = None,
    model_card_markdown: str | None = None,
) -> TrainedModel:
    record = db.get(TrainedModel, registry_id)
    if not record:
        raise ValueError("Registry entry not found")
    if display_name is not None:
        record.display_name = display_name
    if description is not None:
        record.description = description
    if benchmark is not None:
        record.benchmark_json = json.dumps(benchmark)
        if record.lifecycle_state != STATE_RETIRED:
            record.lifecycle_state = STATE_BENCHMARKED
    if model_card_markdown is not None:
        record.model_card_markdown = model_card_markdown
    record.updated_at = datetime.now(timezone.utc)
    job = db.get(TrainingJob, record.job_id)
    if model_card_markdown is None and (description is not None or benchmark is not None):
        record.model_card_markdown = generate_model_card(record, job=job)
    db.commit()
    db.refresh(record)
    return record


def refresh_model_card(db: Session, registry_id: int) -> TrainedModel:
    record = db.get(TrainedModel, registry_id)
    if not record:
        raise ValueError("Registry entry not found")
    job = db.get(TrainingJob, record.job_id)
    record.model_card_markdown = generate_model_card(record, job=job)
    record.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return record


def get_model_by_openai_id(db: Session, model_id: str) -> TrainedModel | None:
    return db.scalar(
        select(TrainedModel).where(TrainedModel.openai_model_id == model_id)
    )


def get_registry_model(db: Session, registry_id: int) -> TrainedModel | None:
    return db.get(TrainedModel, registry_id)


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


def backfill_registry_entries(db: Session) -> int:
    """Populate lifecycle defaults and model cards for DBs created before the registry."""
    updated = 0
    for record in list_models(db):
        changed = False
        if not record.lifecycle_state:
            record.lifecycle_state = "trained"
            changed = True
        if not record.display_name:
            job = db.get(TrainingJob, record.job_id)
            if job:
                record.display_name = job.name
                changed = True
        if not record.metadata_json:
            job = db.get(TrainingJob, record.job_id)
            if job:
                record.metadata_json = json.dumps(
                    build_training_metadata(
                        job=job,
                        config_json=job.config_json,
                        metrics=json.loads(job.metrics_json) if job.metrics_json else None,
                        artifact_path=record.artifact_path,
                        init_mode=None,
                    )
                )
                changed = True
        if not record.model_card_markdown:
            job = db.get(TrainingJob, record.job_id)
            record.model_card_markdown = generate_model_card(record, job=job)
            changed = True
        if changed:
            record.updated_at = datetime.now(timezone.utc)
            updated += 1
    if updated:
        db.commit()
    return updated


def lifecycle_catalog() -> list[dict[str, str]]:
    from app.services.model_lifecycle import ALL_STATES, ALLOWED_TRANSITIONS

    return [
        {
            "state": s,
            "guidance": state_guidance(s),
            "transitions_to": sorted(ALLOWED_TRANSITIONS.get(s, frozenset())),
        }
        for s in sorted(ALL_STATES)
    ]
