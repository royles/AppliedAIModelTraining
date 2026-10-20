"""Ray head / State API defaults."""

from unittest.mock import patch

from app.services.ray_head_options import (
    ray_head_dashboard_cli_args,
    ray_start_include_dashboard,
)


def test_ray_start_include_dashboard_respects_explicit_off(monkeypatch):
    monkeypatch.setenv("RAY_INCLUDE_DASHBOARD", "0")
    assert ray_start_include_dashboard() is False
    assert ray_head_dashboard_cli_args() == ["--include-dashboard=false"]


def test_ray_start_include_dashboard_explicit_on_requires_packages(monkeypatch):
    monkeypatch.setenv("RAY_INCLUDE_DASHBOARD", "1")
    with patch("app.services.ray_head_options.ray_dashboard_importable", return_value=False):
        assert ray_start_include_dashboard() is False
        assert ray_head_dashboard_cli_args() == ["--include-dashboard=false"]
