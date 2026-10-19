"""CDSW worker Ray bootstrap code generation."""

from app.services.cdsw_workers import build_worker_start_code


def test_worker_bootstrap_checks_ray_on_selected_python() -> None:
    code = build_worker_start_code("10.0.0.1:6379", cpus=4, gpus=1)
    assert "importlib.util.find_spec" not in code
    assert "ray_importable(py)" in code
    assert '[py, "-c", "import ray"]' in code
    assert "cannot import ray" in code
