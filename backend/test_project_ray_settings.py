"""Per-project Ray settings are seeded and readable."""

from app.database import SessionLocal, init_db
from app.models import Experiment
from app.services.project_ray_settings import ensure_project_ray_settings, seed_project_ray_settings


def test_project_ray_settings_seeded_on_create():
    init_db()
    db = SessionLocal()
    try:
        exp = Experiment(name="p-ray")
        db.add(exp)
        db.commit()
        db.refresh(exp)
        row = seed_project_ray_settings(db, exp.id)
        assert row.project_id == exp.id
        again = ensure_project_ray_settings(db, exp.id)
        assert again.project_id == exp.id
        assert again.default_job_num_workers >= 1
    finally:
        db.close()
