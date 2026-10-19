"""Memory guidance for Ray / CAI containers (cgroup limits and launch warnings)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# GiB — align with CML resource profiles (profile limit wins over launch_workers memory=).
MIN_SESSION_MEMORY_GB = 8.0
RECOMMENDED_SESSION_MEMORY_GB = 16.0
MIN_WORKER_MEMORY_GB = 8.0
RECOMMENDED_WORKER_MEMORY_GB = 16.0
RECOMMENDED_WORKER_MEMORY_GB_GPU = 32.0

LOW_MEMORY_CONTAINER_GB = 6.0


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
    limit = container_memory_limit_gb()
    return limit is not None and limit < LOW_MEMORY_CONTAINER_GB


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
            "Requesting %.1f GiB per CDSW worker; minimum recommended is %.0f GiB "
            "(profile %s). Ray OOM kills are likely.",
            memory_gb,
            minimum,
            os.environ.get("CML_RESOURCE_PROFILE_ID", "unset"),
        )
    elif memory_gb < recommended:
        logger.info(
            "CDSW worker memory=%.1f GiB (recommended %.0f GiB for this job type).",
            memory_gb,
            recommended,
        )

    session_limit = container_memory_limit_gb()
    if session_limit is not None and session_limit < MIN_SESSION_MEMORY_GB:
        logger.warning(
            "Session container memory limit is ~%.2f GiB; use a >=%.0f GiB session profile "
            "for Ray head + API, and run training on separate worker engines.",
            session_limit,
            RECOMMENDED_SESSION_MEMORY_GB,
        )


def memory_guidance_for_health() -> dict[str, float | str | bool]:
    limit = container_memory_limit_gb()
    return {
        "container_memory_limit_gb": round(limit, 2) if limit is not None else None,
        "low_memory_container": is_low_memory_container(),
        "recommended_session_memory_gb": RECOMMENDED_SESSION_MEMORY_GB,
        "recommended_worker_memory_gb": RECOMMENDED_WORKER_MEMORY_GB,
        "recommended_worker_memory_gb_gpu": RECOMMENDED_WORKER_MEMORY_GB_GPU,
        "min_worker_memory_gb": MIN_WORKER_MEMORY_GB,
    }
