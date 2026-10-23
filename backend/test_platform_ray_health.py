"""Health / platform_ray_status must not call ray.init (GCS log spam on CAI)."""

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


def test_platform_ray_status_uses_ray_status_not_subprocess_init(monkeypatch):
    monkeypatch.setattr(platform_status.cdsw_workers, "cdsw_available", lambda: True)
    calls: list[str] = []

    def _track(name: str):
        def _inner(*_a, **_k):
            calls.append(name)
            if name == "ray_gcs_status_ok":
                return True
            return {"alive": 1.0, "gpu": 1.0, "cpu": 4.0}

        return _inner

    with patch(
        "app.services.platform_status.gcs_tcp_reachable",
        return_value=True,
    ):
        with patch(
            "app.services.ray_gcs_auth.verified_gcs_auth_token",
            return_value="tok",
        ):
            with patch(
                "app.services.ray_cluster_watch.ray_status_cluster_snapshot",
                side_effect=_track("ray_status_cluster_snapshot"),
            ):
                with patch(
                    "app.services.ray_driver_connect.gcs_client_ready",
                    side_effect=AssertionError("health must not call gcs_client_ready"),
                ):
                    out = platform_status.platform_ray_status()
    assert out["state"] == "online"
    assert "ray_status_cluster_snapshot" in calls
