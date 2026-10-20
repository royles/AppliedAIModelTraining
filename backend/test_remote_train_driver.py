import json
import os
import tempfile
from pathlib import Path

from app.services.remote_train_driver import (
    REMOTE_PAYLOAD_NAME,
    use_remote_train_driver,
    write_remote_train_payload,
)


def test_remote_driver_default_when_distributed():
    os.environ.pop("PLATFORM_TRAIN_DRIVER_ON_SESSION", None)
    assert use_remote_train_driver(launch_cdsw_workers=True, pool_is_remote=True)


def test_remote_driver_disabled_on_session_flag():
    os.environ["PLATFORM_TRAIN_DRIVER_ON_SESSION"] = "1"
    try:
        assert not use_remote_train_driver(launch_cdsw_workers=True, pool_is_remote=True)
    finally:
        os.environ.pop("PLATFORM_TRAIN_DRIVER_ON_SESSION", None)


def test_remote_driver_off_without_workers():
    assert not use_remote_train_driver(launch_cdsw_workers=False, pool_is_remote=True)


def test_remote_payload_includes_auth_and_head():
    with tempfile.TemporaryDirectory() as tmp:
        write_remote_train_payload(
            tmp,
            ray_address="10.0.0.1:6379",
            num_workers=2,
            use_gpu=True,
            train_config={"job_id": 1},
            ray_head_address="10.0.0.1:6379",
            ray_auth_token="abc123",
        )
        data = json.loads(Path(tmp, REMOTE_PAYLOAD_NAME).read_text(encoding="utf-8"))
        assert data["ray_auth_token"] == "abc123"
        assert data["ray_head_address"] == "10.0.0.1:6379"
