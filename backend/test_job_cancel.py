"""Stop job validation."""

from app.database import SessionLocal, init_db
from app.models import Experiment, TrainingJob
from app.services.job_cancel import stop_training_job
import pytest


def test_stop_rejects_finished_job():
    init_db()
    db = SessionLocal()
    try:
        exp = Experiment(name="p")
        db.add(exp)
        db.commit()
        job = TrainingJob(
            experiment_id=exp.id,
            name="done",
            status="succeeded",
            model_name="m",
            num_workers=0,
            gpus_per_worker=0,
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            max_steps=10,
        )
        db.add(job)
        db.commit()
        with pytest.raises(ValueError, match="not active"):
            stop_training_job(db, job.id)
    finally:
        db.close()
