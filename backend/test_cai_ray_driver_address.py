"""CAI master session driver must not ray.init via pod IP."""

from unittest.mock import patch

from app.services.cai_networking import (
    coerce_session_driver_ray_address,
    is_cai_master_session,
)


def test_coerce_pod_ip_on_cdsw_home_without_cdsw_module():
    head = "100.100.59.178:6379"
    with patch.dict(
        "os.environ",
        {"HOME": "/home/cdsw", "USER": "cdsw", "RAY_HEAD_ADDRESS": head},
        clear=False,
    ):
        assert is_cai_master_session()
        assert coerce_session_driver_ray_address(head) == "127.0.0.1:6379"


def test_coerce_stale_ray_address_env():
    with patch.dict(
        "os.environ",
        {
            "CDSW_IP_ADDRESS": "100.100.59.178",
            "RAY_ADDRESS": "100.100.59.178:6379",
        },
        clear=False,
    ):
        assert coerce_session_driver_ray_address("100.100.59.178:6379") == "127.0.0.1:6379"
