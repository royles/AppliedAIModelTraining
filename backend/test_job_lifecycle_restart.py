"""Run / Start over must reset Ray cluster and tolerate winding-down threads."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services import job_lifecycle


def test_restart_calls_prepare_before_enqueue(monkeypatch):
    job = MagicMock()
    job.id = 7
    job.status = "stopped"
    job.worker_group = None
    job.experiment_id = 1
    job.ray_address = None
    job.error_message = None
    job.updated_at = None

    db = MagicMock()
    db.get.return_value = job

    prepare = MagicMock()
    monkeypatch.setattr(job_lifecycle, "prepare_job_for_rerun", prepare)
    monkeypatch.setattr(job_lifecycle, "job_is_running", lambda _jid: False)
    monkeypatch.setattr(job_lifecycle, "read_progress", lambda _p: None)
    enqueue = MagicMock()
    monkeypatch.setattr(job_lifecycle, "enqueue_training_job", enqueue)
    monkeypatch.setattr(job_lifecycle, "append_job_log", MagicMock())

    job_lifecycle.restart_training_job(db, 7, resume=False)

    prepare.assert_called_once()
    assert prepare.call_args.kwargs["restart_cluster"] is False
    enqueue.assert_called_once_with(7, resume=False, force_ray_head_restart=True)


def test_restart_propagates_shutdown_wait_error(monkeypatch):
    job = MagicMock()
    job.id = 8
    job.status = "failed"
    job.worker_group = None

    db = MagicMock()
    db.get.return_value = job

    def _fail(_job, **_kwargs):
        raise RuntimeError("Previous run is still shutting down")

    monkeypatch.setattr(job_lifecycle, "prepare_job_for_rerun", _fail)

    with pytest.raises(RuntimeError, match="shutting down"):
        job_lifecycle.restart_training_job(db, 8, resume=True)
