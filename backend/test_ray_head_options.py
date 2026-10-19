"""Ray head / State API defaults."""

from app.services.ray_head_options import ray_start_include_dashboard


def test_ray_start_include_dashboard_respects_explicit_off(monkeypatch):
    monkeypatch.setenv("RAY_INCLUDE_DASHBOARD", "0")
    assert ray_start_include_dashboard() is False


def test_ray_start_include_dashboard_respects_explicit_on(monkeypatch):
    monkeypatch.setenv("RAY_INCLUDE_DASHBOARD", "1")
    assert ray_start_include_dashboard() is True
