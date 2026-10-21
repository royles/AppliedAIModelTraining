"""Lineage snapshots include frozen Ray config and job run details."""

from app.database import SessionLocal, init_db
from app.models import Experiment, TrainingJob, WorkerGroup
from app.services.model_lineage import build_lineage_snapshot
from app.services.project_ray_settings import ensure_project_ray_settings, seed_project_ray_settings


def test_lineage_includes_ray_config_and_job_run():
    init_db()
    db = SessionLocal()
    try:
        exp = Experiment(name="ray-lineage-project")
        db.add(exp)
        db.commit()
        db.refresh(exp)
        seed_project_ray_settings(db, exp.id)
        ray_row = ensure_project_ray_settings(db, exp.id)
        ray_row.gpus_per_worker = 2
        ray_row.worker_cpu = 16.0
        db.commit()

        job = TrainingJob(
            experiment_id=exp.id,
            name="run-1",
            status="succeeded",
            num_workers=4,
            gpus_per_worker=2,
            ray_address="100.64.0.1:6379",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        wg = WorkerGroup(
            job_id=job.id,
            status="stopped",
            target_workers=4,
            head_address="100.64.0.1:6379",
            worker_ids_json='["w1","w2"]',
        )
        db.add(wg)
        db.commit()
        db.refresh(job)

        snap = build_lineage_snapshot(db, job)
        assert snap["ray_config"]["project_id"] == exp.id
        assert snap["ray_config"]["cdsw_workers"]["gpus_per_worker"] == 2
        assert snap["job_run"]["status"] == "succeeded"
        assert snap["job_run"]["ray_address"] == "100.64.0.1:6379"
        assert snap["job_run"]["effective_worker_launch_spec"]["gpus_per_worker"] == 2
        assert snap["job_run"]["worker_group"]["cdsw_worker_ids"] == ["w1", "w2"]
    finally:
        db.close()
