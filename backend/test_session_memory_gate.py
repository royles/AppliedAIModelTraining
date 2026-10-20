"""Session memory gate before Ray Train."""

from app.resource_hints import (
    ray_head_object_store_bytes,
    ray_train_session_memory_error,
)


def test_tiny_session_blocks_ray_train(monkeypatch):
    monkeypatch.setenv("PLATFORM_CAI_LOW_MEMORY", "0")
    monkeypatch.setattr(
        "app.resource_hints.container_memory_limit_gb",
        lambda: 3.66,
    )
    err = ray_train_session_memory_error()
    assert err is not None
    assert "PlacementGroupCleaner" in err


def test_large_session_allows_ray_train(monkeypatch):
    monkeypatch.setattr(
        "app.resource_hints.container_memory_limit_gb",
        lambda: 16.0,
    )
    assert ray_train_session_memory_error() is None


def test_object_store_cap_on_minimum_session(monkeypatch):
    monkeypatch.setattr(
        "app.resource_hints.container_memory_limit_gb",
        lambda: 8.0,
    )
    nbytes = ray_head_object_store_bytes()
    assert nbytes == 512 * 1024 * 1024
