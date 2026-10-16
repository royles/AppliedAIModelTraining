"""Remove training jobs and related records."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.models import TrainedModel, TrainingJob

logger = logging.getLogger(__name__)

def delete_training_job(db: Session, job_id: int) -> None:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise ValueError("Job not found")

    for model in db.scalars(select(TrainedModel).where(TrainedModel.job_id == job_id)):
        db.delete(model)

    if job.worker_group:
        db.delete(job.worker_group)

    db.delete(job)
    db.commit()

    artifact_dir = REPO_ROOT / "data" / "models" / f"job_{job_id}"
    if artifact_dir.is_dir():
        try:
            shutil.rmtree(artifact_dir)
        except OSError as exc:
            logger.warning("Could not remove artifact dir %s: %s", artifact_dir, exc)
