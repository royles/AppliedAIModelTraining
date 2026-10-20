import os
from unittest.mock import patch

from app.services.ray_cai_networking import worker_ray_join_address


def test_worker_ray_join_address_prefers_matching_master_ip():
    with patch.dict(
        os.environ,
        {
            "CDSW_MASTER_IP": "100.100.104.156",
            "PLATFORM_RAY_HEAD_ADDRESS": "100.100.104.156:6379",
        },
        clear=True,
    ):
        assert worker_ray_join_address("100.100.104.156:6379") == "100.100.104.156:6379"


def test_worker_ray_join_address_ignores_stale_master_ip():
    with patch.dict(
        os.environ,
        {
            "CDSW_MASTER_IP": "100.100.59.179",
            "PLATFORM_RAY_HEAD_ADDRESS": "100.100.104.156:6379",
        },
        clear=True,
    ):
        assert worker_ray_join_address("100.100.104.156:6379") == "100.100.104.156:6379"
