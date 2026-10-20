import os
from unittest.mock import patch

from app.services.cai_networking import resolve_ray_gcs_port, resolve_worker_ray_head_address
from app.services.cdsw_worker_codegen import embedded_worker_resolve_head_source
from app.services.cdsw_workers import build_worker_start_code
from app.services.remote_train_driver import build_remote_train_driver_code


def test_resolve_ray_gcs_port_default():
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_ray_gcs_port() == 6379


def test_resolve_worker_head_prefers_cdsw_master_when_matches_pin():
    with patch.dict(
        os.environ,
        {
            "CDSW_MASTER_IP": "100.100.59.154",
            "PLATFORM_RAY_HEAD_ADDRESS": "100.100.59.154:6379",
        },
        clear=True,
    ):
        assert resolve_worker_ray_head_address("100.100.59.154:6379") == "100.100.59.154:6379"


def test_worker_and_remote_driver_share_resolve_head_snippet():
    snippet = embedded_worker_resolve_head_source()
    worker = build_worker_start_code("10.0.0.5:6379")
    remote = build_remote_train_driver_code(1, "10.0.0.5:6379")
    assert snippet.strip() in worker
    assert snippet.strip() in remote
    assert "CDSW_MASTER_IP" in snippet
    assert "PLATFORM_RAY_HEAD_ADDRESS" in snippet
