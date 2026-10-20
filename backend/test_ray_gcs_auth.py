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
        with patch.dict(os.environ, {}, clear=True):
            token = sync_ray_gcs_auth_token("10.0.0.1:6379", required=True)
    assert token == "deadbeef"
    assert path.read_text(encoding="utf-8").strip() == "deadbeef"


def test_invalidate_removes_project_token(tmp_path, monkeypatch):
    path = tmp_path / ".ray_gcs_auth_token"
    path.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.services.ray_gcs_auth.project_gcs_auth_token_path",
        lambda: path,
    )
    invalidate_stored_gcs_auth_tokens()
    assert not path.is_file()
