import os
from pathlib import Path
from unittest.mock import patch

from app.services.ray_gcs_auth import (
    invalidate_stored_gcs_auth_tokens,
    load_gcs_auth_token_from_disk,
    project_gcs_auth_token_path,
    sync_ray_gcs_auth_token,
)


def test_load_token_from_project_file(tmp_path, monkeypatch):
    path = tmp_path / ".ray_gcs_auth_token"
    path.write_text("abc123\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.services.ray_gcs_auth.project_gcs_auth_token_path",
        lambda: path,
    )
    with patch.dict(os.environ, {}, clear=True):
        assert load_gcs_auth_token_from_disk() == "abc123"
        assert os.environ["RAY_AUTH_TOKEN"] == "abc123"


def test_sync_persists_cli_token(tmp_path, monkeypatch):
    path = tmp_path / ".ray_gcs_auth_token"
    monkeypatch.setattr(
        "app.services.ray_gcs_auth.project_gcs_auth_token_path",
        lambda: path,
    )

    class Proc:
        returncode = 0
        stdout = "deadbeef\n"
        stderr = ""

    with patch("app.services.ray_gcs_auth.subprocess.run", return_value=Proc()):
        with patch(
            "app.services.ray_gcs_auth.ray_gcs_status_ok",
            side_effect=lambda _gcs, tok: tok == "deadbeef",
        ):
            with patch.dict(os.environ, {"RAY_AUTH_TOKEN": "stale"}, clear=False):
                token = sync_ray_gcs_auth_token("10.0.0.1:6379", required=True)
    assert token == "deadbeef"
    assert "RAY_AUTH_TOKEN=deadbeef" in path.read_text(encoding="utf-8")
    assert "RAY_GCS_ADDRESS=10.0.0.1:6379" in path.read_text(encoding="utf-8")


def test_sync_strips_stale_env_before_cli(tmp_path, monkeypatch):
    path = tmp_path / ".ray_gcs_auth_token"
    monkeypatch.setattr(
        "app.services.ray_gcs_auth.project_gcs_auth_token_path",
        lambda: path,
    )
    seen_envs: list[dict[str, str]] = []

    class Proc:
        returncode = 0
        stdout = "fresh\n"
        stderr = ""

    def fake_run(_argv, env, **_kwargs):
        seen_envs.append(dict(env))
        return Proc()

    with patch("app.services.ray_gcs_auth.subprocess.run", side_effect=fake_run):
        with patch("app.services.ray_gcs_auth.ray_gcs_status_ok", return_value=False):
            with patch.dict(os.environ, {"RAY_AUTH_TOKEN": "stale"}, clear=False):
                token = sync_ray_gcs_auth_token("10.0.0.1:6379", required=True)
    assert token == "fresh"
    assert seen_envs
    assert "RAY_AUTH_TOKEN" not in seen_envs[0]


def test_invalidate_removes_project_token(tmp_path, monkeypatch):
    path = tmp_path / ".ray_gcs_auth_token"
    path.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.services.ray_gcs_auth.project_gcs_auth_token_path",
        lambda: path,
    )
    invalidate_stored_gcs_auth_tokens()
    assert not path.is_file()
