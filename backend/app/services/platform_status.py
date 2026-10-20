"""Lightweight platform status for the UX header (health poll)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TrainingJob
from app.services import cdsw_workers
from app.services.cai_networking import (
    gcs_tcp_reachable,
    parse_gcs_host_port,
    ray_driver_gcs_address,
    ray_worker_head_address,
)

ACTIVE_JOB_STATUSES = frozenset({"queued", "provisioning", "running"})

_ray_status_cache: dict[str, Any] = {}
_ray_status_cache_lock = threading.Lock()
_RAY_STATUS_CACHE_TTL_SEC = 6.0


def active_jobs_summary(db: Session) -> dict[str, Any]:
    jobs = list(
        db.scalars(
            select(TrainingJob)
            .where(TrainingJob.status.in_(tuple(ACTIVE_JOB_STATUSES)))
            .order_by(TrainingJob.updated_at.desc())
        )
    )
    return {
        "count": len(jobs),
        "jobs": [
            {"id": j.id, "name": j.name, "status": j.status}
            for j in jobs[:8]
        ],
    }


def platform_ray_status() -> dict[str, Any]:
    """Ray GCS reachability without ray.init in the API process."""
    head = ray_worker_head_address()
    gcs = ray_driver_gcs_address(head) if head else None
    if not gcs:
        gcs = os.environ.get("RAY_HEAD_ADDRESS") or os.environ.get("RAY_ADDRESS")
    out: dict[str, Any] = {
        "gcs_address": gcs,
        "cdsw_available": cdsw_workers.cdsw_available(),
        "gcs_reachable": False,
        "alive_nodes": None,
        "gpu_capacity": None,
        "state": "unknown",
    }
    if not gcs:
        out["state"] = "not_configured"
        return out

    if not cdsw_workers.cdsw_available():
        out["state"] = "local"
        return out

    host, port = parse_gcs_host_port(gcs)
    if not gcs_tcp_reachable(host, port, timeout_sec=1.5):
        out["state"] = "deferred"
        return out

    from app.services.ray_gcs_auth import load_gcs_auth_token_from_disk
    from app.services.ray_cluster_watch import ray_status_cluster_snapshot
    from app.services.ray_driver_connect import gcs_client_ready

    token = load_gcs_auth_token_from_disk()
    reachable = gcs_client_ready(gcs, auth_token=token, rpc_timeout_sec=4.0)
    out["gcs_reachable"] = reachable
    if not reachable:
        out["state"] = "deferred"
        return out

    snap = ray_status_cluster_snapshot(gcs, auth_token=token, timeout_sec=6.0)
    if snap is None:
        out["state"] = "reachable"
        return out

    alive = int(snap.get("alive", 0))
    gpu = float(snap.get("gpu", 0.0))
    out["alive_nodes"] = alive
    out["gpu_capacity"] = gpu
    out["state"] = "online" if alive >= 1 else "reachable"
    return out


def cached_platform_ray_status(*, active_job_count: int = 0) -> dict[str, Any]:
    """TTL cache so /api/health polling does not spawn Ray clients every few seconds."""
    if active_job_count > 0:
        gcs = (
            ray_worker_head_address()
            or os.environ.get("RAY_HEAD_ADDRESS")
            or os.environ.get("RAY_ADDRESS")
        )
        return {
            "gcs_address": gcs,
            "cdsw_available": cdsw_workers.cdsw_available(),
            "gcs_reachable": None,
            "alive_nodes": None,
            "gpu_capacity": None,
            "state": "provisioning",
        }

    now = time.monotonic()
    with _ray_status_cache_lock:
        expires = float(_ray_status_cache.get("expires") or 0)
        if expires > now and _ray_status_cache.get("value") is not None:
            return dict(_ray_status_cache["value"])
    value = platform_ray_status()
    with _ray_status_cache_lock:
        _ray_status_cache["value"] = value
        _ray_status_cache["expires"] = now + _RAY_STATUS_CACHE_TTL_SEC
    return dict(value)
