"""Structured job logs for the UX (framework events + training steps)."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.database import SessionLocal
from app.models import TrainingJob
from app.resource_hints import effective_ray_train_workers
from app.services.distributed_training_status import build_distributed_training_status
from app.services.project_ray_settings import ensure_project_ray_settings
from app.services.ray_cluster_topology import (
    build_ray_cluster_topology,
    merge_topology_into_ray_status,
)

LogChannel = Literal["framework", "training"]

_MAX_STORED_LINES = 400
_ACTIVE_LOG_CACHE_SEC = 28.0
_IDLE_LOG_CACHE_SEC = 30.0
_job_logs_cache: dict[int, tuple[float, str, dict[str, Any]]] = {}
_job_logs_cache_lock = threading.Lock()
_ACTIVE_JOB_STATUSES = frozenset({"queued", "provisioning", "running"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def invalidate_job_logs_cache(job_id: int) -> None:
    with _job_logs_cache_lock:
        _job_logs_cache.pop(job_id, None)


def clear_job_run_logs(db: Session, job_id: int) -> None:
    """Remove framework + training log files for a fresh run (UX log panel)."""
    job = db.get(TrainingJob, job_id)
    if not job:
        return
    job.log_json = None
    db.commit()
    path = training_log_path(job_id)
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            path.write_text("", encoding="utf-8")
    invalidate_job_logs_cache(job_id)


def append_job_log(db: Session, job_id: int, channel: LogChannel, message: str) -> None:
    job = db.get(TrainingJob, job_id)
    if not job:
        return
    lines: list[dict[str, str]] = []
    if job.log_json:
        try:
            lines = json.loads(job.log_json)
        except json.JSONDecodeError:
            lines = []
    lines.append({"ts": _now_iso(), "channel": channel, "message": message})
    job.log_json = json.dumps(lines[-_MAX_STORED_LINES:])
    db.commit()


def append_job_log_isolated(job_id: int, channel: LogChannel, message: str) -> None:
    """Commit a log line without holding the caller's session (long Ray / Train waits)."""
    db = SessionLocal()
    try:
        append_job_log(db, job_id, channel, message)
    finally:
        db.close()


def training_log_path(job_id: int) -> Path:
    return REPO_ROOT / "data" / "models" / f"job_{job_id}" / "training.log"


def append_training_log_file(job_id: int, message: str) -> None:
    path = training_log_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(f"{_now_iso()} {message}\n")


def read_training_log_file(job_id: int, tail: int = 200) -> list[dict[str, str]]:
    path = training_log_path(job_id)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, str]] = []
    for raw in lines[-tail:]:
        if not raw.strip():
            continue
        ts, msg = raw.split(" ", 1) if " " in raw else (_now_iso(), raw)
        out.append({"ts": ts, "message": msg})
    return out


def ray_status_snapshot() -> dict[str, Any]:
    try:
        import ray

        initialized = ray.is_initialized()
        status: dict[str, Any] = {
            "initialized": initialized,
            "runtime": "initialized" if initialized else "not_initialized",
        }
        if initialized:
            try:
                status["nodes"] = len(ray.nodes())
                status["cluster_resources"] = ray.cluster_resources()
            except Exception as exc:  # noqa: BLE001
                status["cluster_error"] = str(exc)
        return status
    except ImportError:
        return {"initialized": False, "runtime": "ray_not_installed"}


def _job_logs_cache_key(job: TrainingJob, job_id: int) -> str:
    training_path = training_log_path(job_id)
    training_mtime = ""
    if training_path.is_file():
        try:
            training_mtime = str(int(training_path.stat().st_mtime))
        except OSError:
            training_mtime = ""
    return "|".join(
        (
            job.status,
            str(job.updated_at or ""),
            job.log_json or "",
            job.metrics_json or "",
            job.ray_address or "",
            training_mtime,
        )
    )


def build_job_logs(db: Session, job_id: int) -> dict[str, Any]:
    job = db.get(TrainingJob, job_id)
    if not job:
        return {}

    cache_key = _job_logs_cache_key(job, job_id)
    ttl = _ACTIVE_LOG_CACHE_SEC if job.status in _ACTIVE_JOB_STATUSES else _IDLE_LOG_CACHE_SEC
    now = time.monotonic()
    with _job_logs_cache_lock:
        entry = _job_logs_cache.get(job_id)
        if entry and entry[0] > now and entry[1] == cache_key:
            return dict(entry[2])
    framework: list[dict[str, str]] = []
    if job.log_json:
        try:
            stored = json.loads(job.log_json)
            framework = [e for e in stored if e.get("channel") == "framework"]
        except json.JSONDecodeError:
            framework = []
    training = read_training_log_file(job_id)
    if job.error_message:
        framework.append(
            {"ts": _now_iso(), "message": f"Error: {job.error_message}"},
        )
    ray_status = ray_status_snapshot()
    ray_status["job_ray_address"] = job.ray_address

    expected_workers = job.num_workers or 0
    cdsw_ids: list[str] = []
    if job.worker_group and job.worker_group.worker_ids_json:
        try:
            parsed = json.loads(job.worker_group.worker_ids_json)
            if isinstance(parsed, list):
                cdsw_ids = [str(x) for x in parsed]
        except json.JSONDecodeError:
            cdsw_ids = []

    topology = build_ray_cluster_topology(
        ray_address=job.ray_address,
        job_status=job.status,
        expected_worker_nodes=expected_workers,
        cdsw_worker_ids=cdsw_ids,
    )
    ray_status = merge_topology_into_ray_status(ray_status, topology)

    ray_config = ensure_project_ray_settings(db, job.experiment_id)
    effective_train_workers = effective_ray_train_workers(
        max(1, job.num_workers),
        ray_config.ray_train_num_workers,
        cdsw_worker_engines=expected_workers,
    )
    ray_status["distributed"] = build_distributed_training_status(
        job_id=job_id,
        job_status=job.status,
        num_cdsw_workers=expected_workers,
        ray_train_workers=effective_train_workers,
        gpus_per_worker=job.gpus_per_worker,
        training_log=training,
    )
    topology["distributed"] = ray_status["distributed"]
    for rank_info in ray_status["distributed"].get("ranks") or []:
        if not isinstance(rank_info, dict):
            continue
        rid = int(rank_info.get("rank", 0))
        comp_id = f"train-rank-{rid}"
        topology["components"].append(
            {
                "id": comp_id,
                "role": "train_worker",
                "title": f"Train rank {rank_info.get('display_rank', rid + 1)}",
                "subtitle": str(rank_info.get("device") or "device"),
                "healthy": True,
                "host": str(rank_info.get("host") or ""),
                "metrics": [
                    {
                        "key": "device",
                        "label": "Device",
                        "value": 0,
                        "display": str(rank_info.get("device") or "—"),
                        "unit": "text",
                    },
                    {
                        "key": "world",
                        "label": "World size",
                        "value": float(rank_info.get("world_size") or 0),
                        "display": str(rank_info.get("world_size") or "—"),
                        "unit": "count",
                    },
                ],
            }
        )
        topology["links"].append(
            {
                "from": "train-controller",
                "to": comp_id,
                "healthy": True,
                "kind": "distributed_train",
                "detail": "Ray Train worker process",
            }
        )
    ray_status["topology"] = topology

    if job.status in ("provisioning", "running") and not ray_status.get("initialized"):
        if not topology.get("gcs_reachable"):
            ray_status["note"] = (
                "Cannot reach Ray GCS from the UI snapshot. Confirm scripts/start.py "
                "started the head and the job ray_address is set."
            )
        elif not ray_status.get("note"):
            ray_status["note"] = (
                "Live cluster map below connects to the job Ray address each refresh."
            )

    payload = {
        "job_id": job_id,
        "status": job.status,
        "ray_address": job.ray_address,
        "framework": framework,
        "training": training,
        "ray_status": ray_status,
        "metrics": json.loads(job.metrics_json) if job.metrics_json else None,
    }
    with _job_logs_cache_lock:
        _job_logs_cache[job_id] = (now + ttl, cache_key, dict(payload))
    return payload
