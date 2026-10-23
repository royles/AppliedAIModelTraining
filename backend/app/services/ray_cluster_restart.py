"""Full Ray cluster reset before (re)starting a training run on CAI."""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

from app.models import TrainingJob
from app.services.cai_networking import master_engine_ip
from app.services.cdsw_workers import _import_workers_client
from app.services.job_runner import job_is_running, request_training_job_cancel, stop_active_job_workers
from app.services.log_flair import framework_line

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

_WAIT_FOR_THREAD_SEC = 45.0
_WAIT_POLL_SEC = 0.5


def stop_cdsw_workers_for_job(job: TrainingJob) -> None:
    """Stop in-process pool workers and any CDSW engines recorded on the job."""
    stop_active_job_workers(job.id)
    wg = job.worker_group
    if not wg or not wg.worker_ids_json:
        return
    try:
        parsed = json.loads(wg.worker_ids_json)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed, list) or not parsed:
        return
    client = _import_workers_client()
    if client is None:
        return
    numeric = [int(w) for w in parsed if str(w).isdigit()]
    try:
        if numeric:
            client.stop_workers(*numeric)
        else:
            client.stop_workers()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stop CDSW workers for job %s", job.id)


def full_ray_cluster_restart(*, framework_log: LogFn | None = None) -> str | None:
    """
    ``ray stop --force``, fresh head on ``CDSW_IP_ADDRESS:6379``, refreshed GCS auth.

    No-op when not on a CAI master session (local dev).
    """
    if not master_engine_ip():
        return None

    def _log(msg: str) -> None:
        line = msg if msg.startswith("[") else framework_line("ray", msg)
        if framework_log:
            framework_log(line)
        logger.info(msg)

    from app.services.ray_cai_networking import ensure_ray_head_for_workers

    _log("Full Ray cluster restart (stop workers’ GCS head, ray stop --force, ray start --head)…")
    head = ensure_ray_head_for_workers(framework_log=_log, force_restart=True)
    _log(f"Ray cluster ready at {head}")
    return head


def wait_for_job_thread_to_finish(job_id: int, *, timeout_sec: float = _WAIT_FOR_THREAD_SEC) -> bool:
    """Return True when no training thread is alive for ``job_id``."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not job_is_running(job_id):
            return True
        time.sleep(_WAIT_POLL_SEC)
    return not job_is_running(job_id)


def prepare_job_for_rerun(
    job: TrainingJob,
    *,
    framework_log: LogFn | None = None,
    restart_cluster: bool = True,
) -> None:
    """
    Stop CDSW workers, cancel a winding-down training thread, and restart Ray head.
    """
    job_id = job.id
    if job_is_running(job_id):
        request_training_job_cancel(job_id)
        stop_cdsw_workers_for_job(job)
        if not wait_for_job_thread_to_finish(job_id):
            raise RuntimeError(
                "Previous run is still shutting down on the session. "
                "Wait a few seconds and try Run again."
            )
    else:
        stop_cdsw_workers_for_job(job)

    if restart_cluster:
        full_ray_cluster_restart(framework_log=framework_log)
