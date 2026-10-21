"""Resolve CDSW/CML worker CPU, memory, and GPU for a training job."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.models import PlatformSettings, TrainingJob

logger = logging.getLogger(__name__)


def _worker_resources_from_config(config_json: str | None) -> dict[str, float]:
    if not config_json:
        return {}
    try:
        data = json.loads(config_json)
    except json.JSONDecodeError:
        return {}
    raw = data.get("worker_resources")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    if raw.get("cpu") is not None:
        out["cpu"] = float(raw["cpu"])
    if raw.get("memory_gb") is not None:
        out["memory_gb"] = float(raw["memory_gb"])
    return out


def merge_worker_resources_into_config_json(
    config_json: str | None,
    *,
    worker_cpu: float | None,
    worker_memory_gb: float | None,
) -> str:
    """Persist optional per-job worker sizing inside config_json."""
    if worker_cpu is None and worker_memory_gb is None:
        return config_json or "{}"
    try:
        data = json.loads(config_json) if config_json else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    wr: dict[str, Any] = dict(data.get("worker_resources") or {})
    if worker_cpu is not None:
        wr["cpu"] = float(worker_cpu)
    if worker_memory_gb is not None:
        wr["memory_gb"] = float(worker_memory_gb)
    data["worker_resources"] = wr
    return json.dumps(data)


def resolve_worker_launch_spec(job: TrainingJob, platform: PlatformSettings) -> dict[str, float | int]:
    """Effective launch_workers cpu/memory/gpus for Ray GPU engines on this job."""
    overrides = _worker_resources_from_config(job.config_json)
    cpu = overrides.get("cpu", float(platform.worker_cpu))
    memory_gb = overrides.get("memory_gb", float(platform.worker_memory_gb))
    gpus = job.gpus_per_worker
    if gpus <= 0 and platform.gpus_per_worker > 0:
        gpus = platform.gpus_per_worker
    return {
        "cpu": cpu,
        "memory_gb": memory_gb,
        "gpus_per_worker": int(gpus),
    }


def remote_driver_launch_spec(worker_spec: dict[str, float | int]) -> dict[str, float | int]:
    """CDSW engine for remote Ray Train driver — sized from worker profile, no GPU."""
    worker_cpu = float(worker_spec["cpu"])
    worker_mem = float(worker_spec["memory_gb"])
    return {
        "cpu": max(2.0, min(worker_cpu, 16.0)),
        "memory_gb": max(16.0, min(worker_mem * 0.5, 256.0)),
        "gpus_per_worker": 0,
    }


def format_worker_spec_line(spec: dict[str, float | int], *, num_workers: int) -> str:
    return (
        f"{num_workers}× CDSW engine(s): {spec['cpu']} CPU, {spec['memory_gb']} GiB RAM, "
        f"{spec['gpus_per_worker']} GPU/engine"
    )
