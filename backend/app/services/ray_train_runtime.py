"""Ray Train runtime tuning for CAI (slow workers, state API noise)."""

from __future__ import annotations

import os

from app.services.cdsw_workers import cdsw_available
from app.services.nccl_env import apply_nccl_env


def apply_ray_train_runtime_env() -> None:
    """
    Set driver-side env before Ray Train starts.

    Does not override variables the user already set.
    """
    from app.services.ray_head_options import state_api_reachable
    from app.services.ray_log_noise import apply_ray_train_log_filters

    apply_ray_train_log_filters()

    if not state_api_reachable():
        import logging

        logging.getLogger(__name__).warning(
            "Ray State API is not reachable (dashboard/API server). "
            "PlacementGroupCleaner may log warnings until the head is restarted with "
            "the API enabled (default when ray[default] is installed). "
            "Run: ray stop, then restart scripts/start.py."
        )

    defaults: dict[str, str] = {
        # CDSW workers often need several minutes for pip + ray start --block.
        "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "600",
        "RAY_TRAIN_HEALTH_CHECK_INTERVAL_S": "5",
        # Reduce collective wait warnings if a rank is slow to report (we report on all ranks).
        "RAY_TRAIN_COLLECTIVE_WARN_INTERVAL_S": "600",
    }
    if cdsw_available():
        defaults["RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S"] = "900"
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
    apply_nccl_env()


def cluster_resource_summary() -> dict[str, float]:
    import ray

    if not ray.is_initialized():
        return {}
    try:
        raw = ray.cluster_resources()
        return {str(k): float(v) for k, v in raw.items()}
    except Exception:  # noqa: BLE001
        return {}


def verify_train_resources(
    *,
    num_workers: int,
    use_gpu: bool,
    cpus_per_worker: int = 1,
    gpus_per_worker: int = 1,
) -> None:
    """Fail fast with a clear message when the cluster cannot schedule Train workers."""
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized before Train scheduling")

    resources = cluster_resource_summary()
    if not resources:
        raise RuntimeError("Ray cluster_resources() is empty — workers may have disconnected")

    cpu_need = float(num_workers * max(1, cpus_per_worker))
    cpu_avail = float(resources.get("CPU", 0))
    if cpu_avail < cpu_need:
        raise RuntimeError(
            f"Cluster has {cpu_avail:g} CPU but Ray Train needs {cpu_need:g} "
            f"({num_workers} worker(s) × {cpus_per_worker} CPU). "
            "Check CDSW worker engines are still running and joined the cluster."
        )

    if use_gpu:
        gpu_need = float(num_workers * max(1, gpus_per_worker))
        gpu_avail = float(resources.get("GPU", 0))
        if gpu_avail < gpu_need:
            raise RuntimeError(
                f"Cluster has {gpu_avail:g} GPU but Ray Train needs {gpu_need:g} "
                f"({num_workers} worker(s) × {gpus_per_worker} GPU). "
                "Workers may have exited (see worker logs / ray_process_exit.log) or "
                "ray start did not advertise GPUs — confirm GPU profile and "
                "CML_ACCELERATOR_LABEL_ID."
            )
