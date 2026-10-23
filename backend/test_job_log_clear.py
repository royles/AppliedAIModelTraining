"""Clear stored logs when a task is run again."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services import job_lifecycle
from app.services.job_log import clear_job_run_logs, training_log_path


def test_clear_job_run_logs_wipes_framework_and_training(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.job_log.REPO_ROOT", tmp_path)
    db = MagicMock()
    job = MagicMock()
    job.log_json = '[{"ts":"t","channel":"framework","message":"old"}]'
    db.get.return_value = job

    log_path = training_log_path(3)
    log_path.parent.mkdir(parents=True)
    log_path.write_text("2020-01-01 old training line\n", encoding="utf-8")

    clear_job_run_logs(db, 3)

    assert job.log_json is None
    db.commit.assert_called()
    assert not log_path.is_file()


def test_restart_clears_logs_before_queue_line(monkeypatch, tmp_path):
    monkeypatch.setattr(job_lifecycle, "REPO_ROOT", tmp_path)
    db = MagicMock()
    job = MagicMock()
    job.id = 9
    job.status = "succeeded"
    job.worker_group = None
    job.error_message = None
    job.ray_address = "100.0.0.1:6379"
    db.get.return_value = job

    clear_mock = MagicMock()
    append_mock = MagicMock()
    monkeypatch.setattr(job_lifecycle, "clear_job_run_logs", clear_mock)
    monkeypatch.setattr(job_lifecycle, "append_job_log", append_mock)
    monkeypatch.setattr(job_lifecycle, "prepare_job_for_rerun", lambda *a, **k: None)
    monkeypatch.setattr(job_lifecycle, "read_progress", lambda _p: None)
    monkeypatch.setattr(job_lifecycle, "enqueue_training_job", lambda *a, **k: None)

    job_lifecycle.restart_training_job(db, 9, resume=False)

    clear_mock.assert_called_once_with(db, 9)
    assert append_mock.call_count == 1
