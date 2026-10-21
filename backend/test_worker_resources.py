"""Worker sizing resolution."""

from app.models import PlatformSettings, TrainingJob
from app.services.worker_resources import (
    merge_worker_resources_into_config_json,
    remote_driver_launch_spec,
    resolve_worker_launch_spec,
)


def test_resolve_worker_launch_spec_uses_job_override():
    platform = PlatformSettings(id=1, worker_cpu=4.0, worker_memory_gb=8.0, gpus_per_worker=1)
    job = TrainingJob(
        experiment_id=1,
        name="j",
        num_workers=4,
        gpus_per_worker=1,
        config_json=merge_worker_resources_into_config_json(
            "{}",
            worker_cpu=32.0,
            worker_memory_gb=128.0,
        ),
    )
    spec = resolve_worker_launch_spec(job, platform)
    assert spec["cpu"] == 32.0
    assert spec["memory_gb"] == 128.0
    assert spec["gpus_per_worker"] == 1


def test_remote_driver_scales_with_worker_memory():
    spec = {"cpu": 32.0, "memory_gb": 128.0, "gpus_per_worker": 1}
    driver = remote_driver_launch_spec(spec)
    assert driver["memory_gb"] >= 64.0
    assert driver["gpus_per_worker"] == 0
