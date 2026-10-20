from app.services.ray_memory_config import (
    object_store_bytes_for_instance,
    ray_node_memory_bytes_for_instance,
    ray_start_cli_memory_args,
)


def test_eight_gb_profile_gets_capped_ray_memory():
    args = ray_start_cli_memory_args(8.0)
    assert any(a.startswith("--object-store-memory=") for a in args)
    assert any(a.startswith("--memory=") for a in args)
    assert object_store_bytes_for_instance(8.0) == 512 * 1024 * 1024
    node = ray_node_memory_bytes_for_instance(8.0)
    assert node is not None
    assert 5 * 1024**3 <= node <= 7 * 1024**3


def test_sixteen_gb_profile_uses_ray_defaults():
    assert object_store_bytes_for_instance(16.0) is None
    assert ray_start_cli_memory_args(16.0) == []
