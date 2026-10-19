"""Job state when the framework restarts and user-initiated restart."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.models import TrainingJob
from app.services.job_log import append_job_log
from app.services.job_runner import enqueue_training_job, job_is_running
from app.services.log_flair import framework_line
from app.services.job_progress import read_progress

logger = logging.getLogger(__name__)

_ACTIVE_JOB_STATUSES = frozenset({"queued", "provisioning", "running"})


def reconcile_stale_jobs(db: Session, *, reason: str) -> int:
    """
    Mark in-flight jobs as failed after the API/training process was restarted.

    Training runs in daemon threads; they do not survive a framework restart.
    """
    jobs = list(
        db.scalars(select(TrainingJob).where(TrainingJob.status.in_(_ACTIVE_JOB_STATUSES)))
    )
    if not jobs:
        return 0
    now = datetime.now(timezone.utc)
    for job in jobs:
        job.status = "failed"
        job.error_message = reason
        job.updated_at = now
        if job.worker_group:
            job.worker_group.status = "stopped"
        append_job_log(
            db,
            job.id,
            "framework",
            framework_line("fail", reason),
        )
    db.commit()
    logger.info("Marked %s stale job(s) as failed after framework start", len(jobs))
    return len(jobs)


def restart_training_job(db: Session, job_id: int, *, resume: bool = True) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise ValueError("Job not found")
    if job_is_running(job_id):
        raise RuntimeError("Job is already running in this framework process")
    if job.status != "failed":
        raise ValueError(
            f"Only failed tasks can be restarted (current status={job.status}). "
            "If the framework was restarted, refresh the task list — interrupted runs "
            "are marked failed automatically."
        )

    artifact_dir = (REPO_ROOT / "data" / "models" / f"job_{job_id}").resolve()
    progress = read_progress(artifact_dir) if resume else None
    will_resume = bool(resume and progress and int(progress.get("global_step") or 0) > 0)

    job.status = "queued"
    job.error_message = None
    job.ray_address = None
    job.updated_at = datetime.now(timezone.utc)
    if job.worker_group:
        db.delete(job.worker_group)
    db.commit()
    db.refresh(job)

    if will_resume:
        step = int(progress.get("global_step") or 0)
        epoch = int(progress.get("epoch") or 0)
        append_job_log(
            db,
            job_id,
            "framework",
            framework_line(
                "queue",
                f"Job restart (resume): continuing from step {step} after epoch {epoch}",
            ),
        )
    else:
        append_job_log(
            db,
            job_id,
            "framework",
            framework_line("queue", "Job restart: same parameters, fresh training run"),
        )

    enqueue_training_job(job_id, resume=resume)
    db.refresh(job)
    return job
