from app.services.task_blueprint_store import list_named_blueprints, read_draft, save_named, write_draft


def test_draft_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.task_blueprint_store.BLUEPRINTS_DIR",
        tmp_path,
    )
    written = write_draft({"preset": "scratch_gpt2_tiny", "job_name": "test"})
    assert written.get("kind") == "model_task_blueprint"
    loaded = read_draft()
    assert loaded is not None
    assert loaded["job_name"] == "test"


def test_save_named_list(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.task_blueprint_store.BLUEPRINTS_DIR",
        tmp_path,
    )
    save_named("My Blueprint", {"preset": "x", "job_name": "j"})
    names = list_named_blueprints()
    assert len(names) == 1
    assert names[0]["name"] == "My Blueprint"
