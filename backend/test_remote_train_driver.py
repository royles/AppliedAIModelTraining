import os

from app.services.remote_train_driver import use_remote_train_driver


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
