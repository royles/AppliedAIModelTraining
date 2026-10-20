"""CAI master session driver GCS address resolution."""

from unittest.mock import patch

from app.services.cai_networking import (
    coerce_session_driver_ray_address,
    is_cai_master_session,
    worker_ray_head_from_env,
)
from app.services.ray_driver_connect import (
    _init_attempts,
    master_ip_from_context,
    resolve_driver_gcs_address,
)


def test_init_attempts_never_set_temp_dir():
    attempts = _init_attempts("100.0.0.1:6379", "100.0.0.1")
    assert attempts
    for kw in attempts:
        assert "_temp_dir" not in kw


def test_coerce_uses_master_ip_not_loopback():
    head = "100.100.59.178:6379"
    with patch.dict(
        "os.environ",
        {"CDSW_IP_ADDRESS": "100.100.59.178", "RAY_ADDRESS": "127.0.0.1:6379"},
        clear=False,
    ):
        assert is_cai_master_session()
        assert coerce_session_driver_ray_address("127.0.0.1:6379") == head
        assert resolve_driver_gcs_address("127.0.0.1:6379") == head


def test_master_ip_from_worker_head_when_env_missing():
    with patch.dict("os.environ", {}, clear=True):
        assert master_ip_from_context("100.100.59.178:6379") == "100.100.59.178"
        assert (
            resolve_driver_gcs_address("127.0.0.1:6379", worker_head="100.100.59.178:6379")
            == "100.100.59.178:6379"
        )


def test_coerce_pod_ip_on_cdsw_home():
    head = "100.100.59.178:6379"
    with patch.dict(
        "os.environ",
        {"HOME": "/home/cdsw", "USER": "cdsw", "RAY_HEAD_ADDRESS": head},
        clear=False,
    ):
        assert coerce_session_driver_ray_address(head) == head


def test_worker_head_ignores_worker_cdsw_ip_address():
    fallback = "100.100.59.179:6379"
    with patch.dict(
        "os.environ",
        {"CDSW_IP_ADDRESS": "100.100.59.200", "CDSW_MASTER_IP": "100.100.59.179"},
        clear=True,
    ):
        assert worker_ray_head_from_env(fallback) == fallback
    with patch.dict(
        "os.environ",
        {"CDSW_IP_ADDRESS": "100.100.59.200"},
        clear=True,
    ):
        assert worker_ray_head_from_env(fallback) == fallback
