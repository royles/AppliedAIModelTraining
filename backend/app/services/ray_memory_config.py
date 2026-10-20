"""Ray head/worker memory limits aligned to 8 GiB minimum CAI profiles."""

from __future__ import annotations

import os

from app.resource_hints import MIN_SESSION_MEMORY_GB, MIN_WORKER_MEMORY_GB, container_memory_limit_gb

# Profiles at or below this tier get capped object store + advertised Ray memory.
_MINIMUM_TIER_SLACK_GB = 1.0


def minimum_instance_memory_gb() -> float:
    raw = os.environ.get("PLATFORM_MIN_MEMORY_GB", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return MIN_SESSION_MEMORY_GB


def is_minimum_memory_gb(memory_gb: float | None) -> bool:
    if memory_gb is None:
        return False
    floor = minimum_instance_memory_gb()
    return memory_gb <= floor + _MINIMUM_TIER_SLACK_GB


def container_at_minimum_memory() -> bool:
    return is_minimum_memory_gb(container_memory_limit_gb())


def object_store_bytes_for_instance(memory_gb: float | None) -> int | None:
    """Ray plasma object store cap; keep small on 8 GiB nodes."""
    if memory_gb is None:
        return None
    if not is_minimum_memory_gb(memory_gb):
        return None
    # 512 MiB on minimum tier; leaves room for train torch + raylet on 8 GiB.
    return 512 * 1024 * 1024


def ray_node_memory_bytes_for_instance(memory_gb: float | None) -> int | None:
    """Memory Ray advertises/schedules on this node (below cgroup limit)."""
    if memory_gb is None or not is_minimum_memory_gb(memory_gb):
        return None
    reserve_gib = 2.0 if memory_gb <= minimum_instance_memory_gb() + 0.5 else 2.5
    usable = max(2.0, memory_gb - reserve_gib)
    return int(usable * (1024**3))


def ray_start_cli_memory_args(memory_gb: float | None) -> list[str]:
    args: list[str] = []
    obj = object_store_bytes_for_instance(memory_gb)
    if obj is not None:
        args.append(f"--object-store-memory={obj}")
    node_mem = ray_node_memory_bytes_for_instance(memory_gb)
    if node_mem is not None:
        args.append(f"--memory={node_mem}")
    return args


def worker_ray_memory_env(memory_gb: float) -> dict[str, str]:
    env: dict[str, str] = {}
    obj = object_store_bytes_for_instance(memory_gb)
    node = ray_node_memory_bytes_for_instance(memory_gb)
    if obj is not None:
        env["PLATFORM_RAY_OBJECT_STORE_BYTES"] = str(obj)
    if node is not None:
        env["PLATFORM_RAY_NODE_MEMORY_BYTES"] = str(node)
    return env


def apply_ray_memory_runtime_env() -> dict[str, str]:
    """Driver/worker env defaults for tight 8 GiB clusters (does not override user env)."""
    if not container_at_minimum_memory():
        return {}
    return {
        # Fewer idle workers holding RAM on small nodes.
        "RAY_idle_worker_killing_memory_threshold_bytes": str(50 * 1024 * 1024),
    }
