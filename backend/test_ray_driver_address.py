"""Ray driver should use loopback when GCS is on the session."""

from unittest.mock import patch

from app.services.cdsw_workers import ray_driver_init_address


def test_driver_uses_loopback_when_gcs_local():
    head = "100.100.59.178:6379"
    with patch.dict("os.environ", {"CDSW_IP_ADDRESS": "100.100.59.178"}, clear=False):
        with patch("app.services.cdsw_workers._local_gcs_reachable", return_value=True):
            with patch("app.services.cdsw_workers.cdsw_available", return_value=False):
                assert ray_driver_init_address(head) == "127.0.0.1:6379"


def test_resolve_ray_address_rewrites_stale_ray_address_env():
    from app.training.transformer_job import resolve_ray_address

    with patch.dict(
        "os.environ",
        {
            "CDSW_IP_ADDRESS": "100.100.59.178",
            "RAY_ADDRESS": "100.100.59.178:6379",
        },
        clear=False,
    ):
        with patch("app.services.cdsw_workers.cdsw_available", return_value=False):
            assert resolve_ray_address(None) == "127.0.0.1:6379"
