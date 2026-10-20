import os
from unittest.mock import patch

from app.services.ray_cai_networking import (
    build_worker_connectivity_probe_code,
    describe_worker_probe_failure,
    worker_connectivity_probe_passed,
    worker_ray_join_address,
)


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


def test_probe_code_embeds_auth_and_uses_venv():
    code = build_worker_connectivity_probe_code(
        "100.100.59.154:6379",
        "/tmp/probe.json",
        ray_auth_token="secret-token",
    )
    assert 'INLINE_RAY_AUTH_TOKEN = "secret-token"' in code
    assert ".venv" in code
    assert "ray.scripts.scripts" in code
    assert "ray.init(address=head, ignore_reinit_error=True)" in code


def test_probe_passes_on_ray_status_without_ray_init():
    assert worker_connectivity_probe_passed(
        {"tcp": True, "ray_ok": False, "ray_status_ok": True}
    )


def test_describe_probe_failure_tcp_without_ray_auth():
    probe = {
        "tcp": True,
        "ray_ok": False,
        "auth_configured": False,
        "ray_error": "Failed to connect",
    }
    msg = describe_worker_probe_failure(probe, "100.100.59.154:6379")
    assert "auth token missing" in msg
    assert "100.100.59.154:6379" in msg


def test_describe_probe_failure_no_tcp():
    probe = {"tcp": False, "tcp_error": "timed out"}
    msg = describe_worker_probe_failure(probe, "1.2.3.4:6379")
    assert "cannot open TCP" in msg
