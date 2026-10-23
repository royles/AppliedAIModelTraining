"""Health / platform_ray_status must stay lightweight (no Ray subprocess on poll)."""

from __future__ import annotations

from unittest.mock import patch

from app.services import platform_status


def test_platform_ray_status_deferred_when_tcp_closed(monkeypatch):
    monkeypatch.setattr(platform_status.cdsw_workers, "cdsw_available", lambda: True)
    with patch.dict(
        "os.environ",
        {"CDSW_IP_ADDRESS": "100.100.191.17", "RAY_HEAD_ADDRESS": "100.100.191.17:6379"},
        clear=False,
    ):
        with patch(
            "app.services.platform_status.gcs_tcp_reachable",
            return_value=False,
        ):
            out = platform_status.platform_ray_status()
    assert out["state"] == "deferred"
    assert out["gcs_reachable"] is False


def test_platform_ray_status_tcp_only_no_ray_cli(monkeypatch):
    monkeypatch.setattr(platform_status.cdsw_workers, "cdsw_available", lambda: True)
    with patch(
        "app.services.platform_status.gcs_tcp_reachable",
        return_value=True,
    ):
        with patch(
            "app.services.ray_gcs_auth.verified_gcs_auth_token",
            side_effect=AssertionError("health must not verify GCS auth"),
        ):
            with patch(
                "app.services.ray_cluster_watch.ray_status_cluster_snapshot",
                side_effect=AssertionError("health must not run ray status"),
            ):
                out = platform_status.platform_ray_status()
    assert out["state"] == "reachable"
    assert out["gcs_reachable"] is True
