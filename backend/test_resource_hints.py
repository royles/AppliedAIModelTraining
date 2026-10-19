"""Ray Train worker count vs CDSW engines."""

from app.resource_hints import effective_ray_train_workers


def test_cdsw_jobs_ignore_session_low_memory_cap(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_CAI_LOW_MEMORY", "1")
    assert (
        effective_ray_train_workers(4, persisted_cap=0, cdsw_worker_engines=4) == 4
    )


def test_session_only_job_still_capped_on_low_memory(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_CAI_LOW_MEMORY", "1")
    assert effective_ray_train_workers(4, persisted_cap=0, cdsw_worker_engines=0) == 1


def test_explicit_cap_applies_without_cdsw_workers(monkeypatch) -> None:
    monkeypatch.delenv("RAY_TRAIN_NUM_WORKERS", raising=False)
    assert effective_ray_train_workers(8, persisted_cap=2, cdsw_worker_engines=0) == 2
