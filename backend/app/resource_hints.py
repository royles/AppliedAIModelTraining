"""Memory guidance for Ray / CAI containers (cgroup limits and launch warnings)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# GiB — platform minimum bar (head + workers); match CML resource profiles.
MIN_SESSION_MEMORY_GB = 8.0
RECOMMENDED_SESSION_MEMORY_GB = 16.0
MIN_WORKER_MEMORY_GB = 8.0
RECOMMENDED_WORKER_MEMORY_GB = 16.0
RECOMMENDED_WORKER_MEMORY_GB_GPU = 32.0

# Containers below this use low-memory Train caps (batch size, session-only worker count).
LOW_MEMORY_CONTAINER_GB = MIN_SESSION_MEMORY_GB
MIN_SESSION_MEMORY_FOR_RAY_TRAIN_GB = MIN_SESSION_MEMORY_GB
# 8 GiB CML profiles often report ~7.3–7.9 GiB in cgroup; allow nominal minimum tier.
SESSION_MEMORY_CGROUP_SLACK_GB = 0.85
ABSOLUTE_MIN_SESSION_FOR_RAY_TRAIN_GB = 6.5


def _read_cgroup_limit_bytes() -> int | None:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        raw = path.read_text().strip()
        if not raw or raw == "max":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def container_memory_limit_gb() -> float | None:
    limit = _read_cgroup_limit_bytes()
    if limit is None:
        return None
    return limit / (1024**3)


def is_low_memory_container() -> bool:
    if os.environ.get("PLATFORM_CAI_LOW_MEMORY", "").lower() in ("1", "true", "yes"):
        return True
    from app.services.ray_memory_config import container_at_minimum_memory

    return container_at_minimum_memory()


def should_apply_train_memory_savings(worker_memory_gb: float | None = None) -> bool:
    """Batch caps and similar when session or workers are on the 8 GiB tier."""
    if is_low_memory_container():
        return True
    if worker_memory_gb is None:
        return False
    from app.services.ray_memory_config import is_minimum_memory_gb

    return is_minimum_memory_gb(worker_memory_gb)


def effective_ray_train_workers(
    requested: int,
    persisted_cap: int | None = None,
    *,
    cdsw_worker_engines: int = 0,
) -> int:
    """
    Ray Train worker processes (TorchTrainer world size).

    When CDSW worker engines are provisioned, training MUST use one rank per engine;
    session low-memory caps do not apply (compute runs on worker pods, not the session).
    """
    requested = max(1, requested)
    engines = max(0, int(cdsw_worker_engines))
    if engines > 0:
        return max(requested, engines)

    cap = persisted_cap if persisted_cap is not None else 0
    if cap <= 0:
        cap = int(os.environ.get("RAY_TRAIN_NUM_WORKERS", "0") or 0)
    if cap > 0:
        return min(requested, cap)
    if is_low_memory_container():
        return 1
    return requested


def log_worker_launch_memory(memory_gb: float, gpus: int) -> None:
    minimum = MIN_WORKER_MEMORY_GB
    recommended = RECOMMENDED_WORKER_MEMORY_GB_GPU if gpus > 0 else RECOMMENDED_WORKER_MEMORY_GB
    if memory_gb < minimum:
        logger.warning(
            "Requesting %.1f GiB per CDSW worker; platform minimum is %.0f GiB "
            "(profile %s). Ray OOM kills are likely below the minimum bar.",
            memory_gb,
            minimum,
            os.environ.get("CML_RESOURCE_PROFILE_ID", "unset"),
        )
    elif memory_gb < recommended:
        logger.info(
            "CDSW worker memory=%.1f GiB (minimum bar %.0f GiB; recommended %.0f GiB).",
            memory_gb,
            minimum,
            recommended,
        )

    session_limit = container_memory_limit_gb()
    if session_limit is not None and session_limit < MIN_SESSION_MEMORY_GB:
        logger.warning(
            "Session container memory limit is ~%.2f GiB; use a >=%.0f GiB session profile "
            "for Ray head + API, and run training on separate worker engines.",
            session_limit,
            MIN_SESSION_MEMORY_GB,
        )


def effective_session_memory_floor_gb() -> float:
    from app.services.ray_memory_config import minimum_instance_memory_gb

    nominal = minimum_instance_memory_gb() - SESSION_MEMORY_CGROUP_SLACK_GB
    return max(ABSOLUTE_MIN_SESSION_FOR_RAY_TRAIN_GB, nominal)


def session_memory_too_small_for_ray_train() -> bool:
    limit = container_memory_limit_gb()
    if limit is None:
        return False
    return limit < effective_session_memory_floor_gb()


def ray_train_session_memory_error(
    *,
    cdsw_worker_engines: int = 0,
    non_distributed_low_memory: bool = False,
) -> str | None:
    """
    Block only when Train would run entirely on a tiny session with no worker pool.

    Distributed jobs (CDSW engines > 0) use worker memory; the session holds Ray control only.
    """
    if cdsw_worker_engines > 0:
        return None

    limit = container_memory_limit_gb()
    if limit is None:
        return None

    if non_distributed_low_memory:
        if limit >= ABSOLUTE_MIN_SESSION_FOR_RAY_TRAIN_GB:
            return None
        return (
            f"Session cgroup memory is ~{limit:.2f} GiB — too small even for non-distributed "
            f"low-memory mode (need ~>={ABSOLUTE_MIN_SESSION_FOR_RAY_TRAIN_GB:.1f} GiB). "
            f"Use a larger session profile or enable distributed CDSW workers."
        )

    if not session_memory_too_small_for_ray_train():
        return None
    floor = effective_session_memory_floor_gb()
    return (
        f"Session cgroup memory is ~{limit:.2f} GiB (need ~>={floor:.1f} GiB measured for "
        f"session-only Train, or enable CDSW workers for distributed memory). "
        f"For session-only on a small pod, turn on **Non-distributed low memory mode** in "
        f"Framework Configuration, or upgrade the session profile and run `ray stop` before retrying."
    )


def ray_head_object_store_bytes() -> int | None:
    from app.services.ray_memory_config import object_store_bytes_for_instance

    return object_store_bytes_for_instance(container_memory_limit_gb())


def ray_head_start_memory_args() -> list[str]:
    from app.services.ray_memory_config import ray_start_cli_memory_args

    return ray_start_cli_memory_args(container_memory_limit_gb())


def memory_guidance_for_health() -> dict[str, float | str | bool]:
    from app.services.ray_memory_config import minimum_instance_memory_gb

    limit = container_memory_limit_gb()
    return {
        "container_memory_limit_gb": round(limit, 2) if limit is not None else None,
        "low_memory_container": is_low_memory_container(),
        "minimum_instance_memory_gb": minimum_instance_memory_gb(),
        "recommended_session_memory_gb": RECOMMENDED_SESSION_MEMORY_GB,
        "recommended_worker_memory_gb": RECOMMENDED_WORKER_MEMORY_GB,
        "recommended_worker_memory_gb_gpu": RECOMMENDED_WORKER_MEMORY_GB_GPU,
        "min_worker_memory_gb": MIN_WORKER_MEMORY_GB,
        "min_session_memory_for_ray_train_gb": MIN_SESSION_MEMORY_FOR_RAY_TRAIN_GB,
        "session_too_small_for_ray_train": session_memory_too_small_for_ray_train(),
    }
