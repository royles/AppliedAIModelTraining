"""Ray driver should use master engine IP on CAI (not loopback-only coercion)."""

from unittest.mock import patch

from app.services.cdsw_workers import ray_driver_init_address


def test_driver_uses_master_ip_on_cai():
    head = "100.100.59.178:6379"
    with patch.dict("os.environ", {"CDSW_IP_ADDRESS": "100.100.59.178"}, clear=False):
        with patch("app.services.cdsw_workers.cdsw_available", return_value=False):
            assert ray_driver_init_address(head) == head


def test_resolve_ray_address_rewrites_loopback_to_master():
    from app.training.transformer_job import resolve_ray_address

    with patch.dict(
        "os.environ",
        {
            "CDSW_IP_ADDRESS": "100.100.59.178",
            "RAY_ADDRESS": "127.0.0.1:6379",
        },
        clear=False,
    ):
        with patch("app.services.cdsw_workers.cdsw_available", return_value=False):
            assert resolve_ray_address(None) == "100.100.59.178:6379"
