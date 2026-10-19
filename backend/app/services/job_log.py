"""Structured job logs for the UX (framework events + training steps)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.models import TrainingJob

LogChannel = Literal["framework", "training"]

_MAX_STORED_LINES = 400


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def build_job_logs(db: Session, job_id: int) -> dict[str, Any]:
    job = db.get(TrainingJob, job_id)
    if not job:
        return {}
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
    if job.status in ("provisioning", "running") and not ray_status.get("initialized"):
        ray_status["note"] = (
            "Status reflects the web API process. Training connects to Ray in a "
            "background thread; see framework logs after 'Ray Train job submitted'."
        )

    return {
        "job_id": job_id,
        "status": job.status,
        "ray_address": job.ray_address,
        "framework": framework,
        "training": training,
        "ray_status": ray_status,
        "metrics": json.loads(job.metrics_json) if job.metrics_json else None,
    }
