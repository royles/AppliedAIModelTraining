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
_RAY_STATUS_CACHE_TTL_OK_SEC = 12.0
_RAY_STATUS_CACHE_TTL_DEFERRED_SEC = 45.0


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
    """
    Ray GCS reachability for /api/health without ``ray.init`` in the API process.

    Cloudera CAI: GCS listens on ``CDSW_IP_ADDRESS:6379`` after ``ray start --head``.
    TCP can be open while RPC/auth is not ready — probing with ``ray.init`` every UI
    poll causes log spam (``Failed to connect to GCS … within 5 seconds``). Use
    ``ray status`` only here; full RPC waits belong in job provisioning.
    """
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
    try:
        if not gcs:
            out["state"] = "not_configured"
            return out

        if not cdsw_workers.cdsw_available():
            out["state"] = "local"
            return out

        host, port = parse_gcs_host_port(gcs)
        if not gcs_tcp_reachable(host, port, timeout_sec=1.5):
            out["state"] = "deferred"
            out["detail"] = (
                "Ray head not listening on master pod IP yet (head starts with model tasks). "
                "See docs/CAI_RAY_NETWORKING.md"
            )
            return out

        from app.services.ray_cluster_watch import ray_status_cluster_snapshot
        from app.services.ray_gcs_auth import verified_gcs_auth_token

        token = verified_gcs_auth_token(gcs, timeout_sec=10.0)
        out["gcs_reachable"] = token is not None
        if not token:
            out["state"] = "deferred"
            out["detail"] = (
                "GCS port open but auth/RPC not ready (stale token, head starting, or head down). "
                "Run: ray stop --force; rm -f data/.ray_gcs_auth_token; python3 scripts/start.py. "
                "Workers join via CDSW_MASTER_IP:6379 per Cloudera worker networking."
            )
            return out

        snap = ray_status_cluster_snapshot(gcs, auth_token=token, timeout_sec=12.0)
        if snap is None:
            out["state"] = "reachable"
            return out

        alive = int(snap.get("alive", 0))
        gpu = float(snap.get("gpu", 0.0))
        out["alive_nodes"] = alive
        out["gpu_capacity"] = gpu
        out["state"] = "online" if alive >= 1 else "reachable"
        return out
    except Exception as exc:  # noqa: BLE001
        out["state"] = "deferred"
        out["detail"] = f"platform_ray_status: {exc}"
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
    ttl = (
        _RAY_STATUS_CACHE_TTL_OK_SEC
        if value.get("gcs_reachable")
        else _RAY_STATUS_CACHE_TTL_DEFERRED_SEC
    )
    with _ray_status_cache_lock:
        _ray_status_cache["value"] = value
        _ray_status_cache["expires"] = now + ttl
    return dict(value)
