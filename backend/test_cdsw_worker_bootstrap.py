"""CDSW worker Ray bootstrap code generation."""

from app.services.cdsw_workers import build_worker_start_code


def test_worker_bootstrap_checks_ray_on_selected_python() -> None:
    code = build_worker_start_code("10.0.0.1:6379", cpus=4, gpus=1)
    assert "importlib.util.find_spec" not in code
    assert "ray_importable(py)" in code
    assert '[py, "-c", "import ray"]' in code
    assert "cannot import ray" in code


def test_worker_bootstrap_resolves_head_from_master_ip_not_worker_ip() -> None:
    code = build_worker_start_code("100.100.59.179:6379", cpus=4, gpus=1)
    assert "wait_for_gcs" in code
    assert "FALLBACK_HEAD = \"100.100.59.179:6379\"" in code
    assert "PLATFORM_RAY_HEAD_ADDRESS" in code
    resolve_body = code.split("def resolve_head_with_meta")[1].split("def wait_for_gcs")[0]
    assert "CDSW_IP_ADDRESS" not in resolve_body
    assert "PLATFORM_RAY_HEAD_ADDRESS" in resolve_body
