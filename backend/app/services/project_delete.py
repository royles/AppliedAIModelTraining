"""Remove training projects and their jobs."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Experiment, TrainingJob
from app.services.job_delete import delete_training_job


def delete_project(db: Session, project_id: int) -> None:
    exp = db.get(Experiment, project_id)
    if not exp:
        raise ValueError("Project not found")

    job_ids = list(
        db.scalars(select(TrainingJob.id).where(TrainingJob.experiment_id == project_id))
    )
    for job_id in job_ids:
        delete_training_job(db, job_id)

    db.delete(exp)
    db.commit()
