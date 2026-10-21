"""User-initiated training job stop."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import TrainingJob
from app.services.job_log import append_job_log
from app.services.job_runner import (
    job_is_running,
    request_training_job_cancel,
    stop_active_job_workers,
)
from app.services.log_flair import framework_line

_ACTIVE_JOB_STATUSES = frozenset({"queued", "provisioning", "running"})


def stop_training_job(db: Session, job_id: int) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise ValueError("Job not found")
    if job.status not in _ACTIVE_JOB_STATUSES:
        raise ValueError(f"Job is not active (status={job.status})")

    request_training_job_cancel(job_id)
    stop_active_job_workers(job_id)

    append_job_log(
        db,
        job_id,
        "framework",
        framework_line("info", "Stop requested — shutting down workers and training"),
    )
    job.status = "stopped"
    job.error_message = "Stopped by user"
    job.updated_at = datetime.now(timezone.utc)
    if job.worker_group:
        job.worker_group.status = "stopped"
    db.commit()
    db.refresh(job)
    if not job_is_running(job_id):
        from app.services.job_runner import clear_training_job_cancel

        clear_training_job_cancel(job_id)
    return job
