import json

from app.database import SessionLocal, init_db
from app.models import Experiment
from app.services.task_versions import content_hash, create_task_with_version, list_versions


def test_task_version_append_only():
    init_db()
    db = SessionLocal()
    try:
        exp = Experiment(name="test-project")
        db.add(exp)
        db.commit()
        db.refresh(exp)
        spec = {"kind": "model_task_blueprint", "schema_version": 1, "preset": "x"}
        task, v1 = create_task_with_version(
            db, project_id=exp.id, task_name="my-task", spec=spec
        )
        assert v1.version == 1
        assert v1.content_hash == content_hash(spec)
        versions = list_versions(db, task.id)
        assert len(versions) == 1
        assert json.loads(versions[0].spec_json)["preset"] == "x"
    finally:
        db.close()
