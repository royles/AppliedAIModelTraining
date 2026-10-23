"""Verified GCS auth must not reuse stale tokens (Ray 2.59+ InvalidAuthToken)."""

from __future__ import annotations

import os
from unittest.mock import patch

from app.services.ray_gcs_auth import invalidate_stored_gcs_auth_tokens, verified_gcs_auth_token


def test_verified_token_syncs_when_env_invalid():
    with patch(
        "app.services.ray_gcs_auth.ray_gcs_status_ok",
        return_value=False,
    ):
        with patch(
            "app.services.ray_gcs_auth.sync_ray_gcs_auth_token",
            return_value="fresh",
        ) as sync:
            with patch.dict(os.environ, {"RAY_AUTH_TOKEN": "stale"}, clear=False):
                assert verified_gcs_auth_token("100.100.191.20:6379") == "fresh"
    sync.assert_called_once()


def test_ray_gcs_status_ok_clears_stale_on_invalid_auth(tmp_path, monkeypatch):
    path = tmp_path / ".ray_gcs_auth_token"
    path.write_text("RAY_GCS_ADDRESS=100.100.191.20:6379\nRAY_AUTH_TOKEN=bad\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.services.ray_gcs_auth.project_gcs_auth_token_path",
        lambda: path,
    )

    class Proc:
        returncode = 1
        stdout = ""
        stderr = "InvalidAuthToken: Authentication token is missing or incorrect"

    with patch("app.services.ray_gcs_auth.subprocess.run", return_value=Proc()):
        with patch.dict(os.environ, {"RAY_AUTH_TOKEN": "bad"}, clear=False):
            from app.services.ray_gcs_auth import ray_gcs_status_ok

            assert ray_gcs_status_ok("100.100.191.20:6379", "bad", timeout_sec=1.0) is False
            assert not path.is_file()
            assert "RAY_AUTH_TOKEN" not in os.environ
