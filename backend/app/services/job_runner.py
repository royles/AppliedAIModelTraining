from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.resource_hints import (
    effective_ray_train_workers,
    ray_train_session_memory_error,
    should_apply_train_memory_savings,
)
from app.database import SessionLocal
from app.models import Experiment, TrainingJob, WorkerGroup
from app.services.cdsw_workers import CdswRayWorkerPool, ray_driver_init_address
from app.services.platform_settings import (
    resolved_ray_head,
    worker_pool_kwargs,
)
from app.services.job_log import append_job_log, append_job_log_isolated
from app.services.log_flair import framework_banner, framework_line
from app.services.model_registry import default_openai_model_id, register_trained_model
from app.training.transformer_config import TransformerTrainingConfig
from app.services.worker_resources import (
    remote_driver_launch_spec,
    resolve_worker_launch_spec,
)

logger = logging.getLogger(__name__)

_job_threads: dict[int, threading.Thread] = {}
_job_thread_lock = threading.Lock()
_pending_resume: dict[int, bool] = {}
_pending_ray_force_head: dict[int, bool] = {}
_job_cancel_requested: set[int] = set()
_job_pools: dict[int, CdswRayWorkerPool] = {}


class JobCancelledError(Exception):
    """Raised when the user stops a job while it is running."""


def request_training_job_cancel(job_id: int) -> None:
    with _job_thread_lock:
        _job_cancel_requested.add(job_id)


def clear_training_job_cancel(job_id: int) -> None:
    with _job_thread_lock:
        _job_cancel_requested.discard(job_id)


def is_training_job_cancel_requested(job_id: int) -> bool:
    with _job_thread_lock:
        return job_id in _job_cancel_requested


def _register_job_pool(job_id: int, pool: CdswRayWorkerPool) -> None:
    with _job_thread_lock:
        _job_pools[job_id] = pool


def _unregister_job_pool(job_id: int) -> None:
    with _job_thread_lock:
        _job_pools.pop(job_id, None)


def stop_active_job_workers(job_id: int) -> None:
    with _job_thread_lock:
        pool = _job_pools.get(job_id)
    if pool is not None:
        try:
            pool.stop_all()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to stop workers for job %s", job_id)


def _raise_if_cancelled(job_id: int) -> None:
    if is_training_job_cancel_requested(job_id):
        raise JobCancelledError("Stopped by user")


def job_is_running(job_id: int) -> bool:
    with _job_thread_lock:
        thread = _job_threads.get(job_id)
        return thread is not None and thread.is_alive()


def _pop_resume_flag(job_id: int) -> bool:
    with _job_thread_lock:
        return _pending_resume.pop(job_id, False)


def _pop_ray_force_head(job_id: int) -> bool:
    with _job_thread_lock:
        return _pending_ray_force_head.pop(job_id, False)


def _update_job(db: Session, job_id: int, **fields) -> None:
    job = db.get(TrainingJob, job_id)
    if not job:
        return
    for key, val in fields.items():
        setattr(job, key, val)
    job.updated_at = datetime.now(timezone.utc)
    db.commit()


def resolve_job_config(job: TrainingJob) -> TransformerTrainingConfig:
    return TransformerTrainingConfig.from_legacy_job_fields(
        model_name=job.model_name,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        max_steps=job.max_steps,
        epochs=job.epochs,
        config_json=job.config_json,
    )


def _release_job_db(db: Session | None) -> None:
    if db is not None:
        db.close()


def _require_job_db(db: Session | None, job_id: int) -> tuple[Session, TrainingJob]:
    session = db if db is not None else SessionLocal()
    job = session.get(TrainingJob, job_id)
    if not job:
        if db is None:
            session.close()
        raise RuntimeError(f"Training job {job_id} not found")
    return session, job


def run_training_job(job_id: int) -> None:
    db: Session | None = SessionLocal()
    pool: CdswRayWorkerPool | None = None
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        if is_training_job_cancel_requested(job_id):
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line("info", "Job stopped before provisioning started"),
            )
            _update_job(db, job_id, status="stopped", error_message="Stopped by user")
            clear_training_job_cancel(job_id)
            return
        _update_job(db, job_id, status="provisioning")
        append_job_log(
            db,
            job_id,
            "framework",
            framework_banner("Model task provisioning"),
        )
        append_job_log(db, job_id, "framework", framework_line("queue", "Job queued"))

        if job.worker_group:
            db.delete(job.worker_group)
            db.commit()

        cfg = resolve_job_config(job)
        artifact_dir = str((REPO_ROOT / "data" / "models" / f"job_{job_id}").resolve())
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        train_config = cfg.to_train_dict(job_id=job_id, artifact_dir=artifact_dir)
        train_config["job_epochs"] = job.epochs

        from app.services.job_progress import apply_resume_to_train_config

        resume_run = _pop_resume_flag(job_id)
        force_ray_head_restart = _pop_ray_force_head(job_id)
        train_config = apply_resume_to_train_config(
            train_config, artifact_dir, resume=resume_run
        )

        from app.training.dataset_staging import (
            attach_shared_collection_to_train_config,
            shared_dataset_dirs_from_config,
        )

        dataset_cfg = train_config.get("dataset") or cfg.dataset.model_dump()
        train_config = attach_shared_collection_to_train_config(
            train_config,
            dataset_cfg=dataset_cfg,
            task_type=str(train_config.get("task_type") or cfg.task_type),
        )
        if train_config.get("dataset_access_mode") == "remote":
            remote_uri = train_config.get("dataset_remote_uri") or "remote catalog"
            registry = train_config.get("data_collection_local_path") or ""
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "train",
                    f"Dataset from remote ({remote_uri})"
                    + (f"; collection registry {registry}" if registry else "")
                    + " — no local corpus copy",
                ),
            )
        else:
            shared_dirs = shared_dataset_dirs_from_config(train_config)
            if shared_dirs:
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "train",
                        f"Using shared data collection on disk: {shared_dirs[0]}",
                    ),
                )

        if train_config.get("resume") and train_config.get("resume_global_step"):
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "train",
                    f"Resuming from step {train_config['resume_global_step']} "
                    f"(checkpoint={train_config.get('resume_checkpoint_path', 'weights only')})",
                ),
            )

        from app.services.project_ray_settings import ensure_project_ray_settings

        platform = ensure_project_ray_settings(db, job.experiment_id)
        from app.services.cai_networking import (
            format_ray_head,
            master_engine_ip,
            parse_gcs_host_port,
            reconcile_live_ray_head_env,
        )

        reconcile_live_ray_head_env()
        head = resolved_ray_head(platform)
        live_ip = master_engine_ip()
        if live_ip and head:
            head_host, _ = parse_gcs_host_port(head)
            if head_host != live_ip:
                live_head = format_ray_head(live_ip)
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "warn",
                        f"Ray head was {head}; using live session master {live_head}",
                    ),
                )
                head = live_head
        gpus_per_worker = job.gpus_per_worker
        if gpus_per_worker <= 0 and platform.gpus_per_worker > 0:
            gpus_per_worker = platform.gpus_per_worker
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "train",
                    f"Job GPUs/worker was 0; using platform default {gpus_per_worker}.",
                ),
            )
        worker_spec = resolve_worker_launch_spec(job, platform)
        pool_kwargs = {
            **worker_pool_kwargs(platform),
            "cpu": float(worker_spec["cpu"]),
            "memory_gb": float(worker_spec["memory_gb"]),
            "gpus_per_worker": int(worker_spec["gpus_per_worker"]),
        }
        pool = CdswRayWorkerPool(**pool_kwargs)
        _register_job_pool(job_id, pool)
        _raise_if_cancelled(job_id)

        needs_cai_ray_head = bool(live_ip) and not (
            platform.ray_local_mode and not pool.is_remote
        )
        if force_ray_head_restart and needs_cai_ray_head:
            from app.services.ray_cluster_restart import full_ray_cluster_restart

            full_ray_cluster_restart(
                framework_log=lambda msg: append_job_log(
                    db,
                    job_id,
                    "framework",
                    msg if msg.startswith("[") else framework_line("ray", msg),
                )
            )
            force_ray_head_restart = False
        from app.services.job_parallel_startup import parallel_dataset_prep_and_ray_head

        try:
            train_config, head = parallel_dataset_prep_and_ray_head(
                job_id=job_id,
                cfg=cfg,
                train_config=train_config,
                needs_cai_ray_head=needs_cai_ray_head,
                force_ray_head_restart=force_ray_head_restart,
                head_before=head,
                resume=bool(train_config.get("resume")),
            )
        except ValueError as exc:
            append_job_log(db, job_id, "framework", f"From-scratch prep failed: {exc}")
            raise

        worker_group = WorkerGroup(
            job_id=job.id,
            target_workers=job.num_workers,
            head_address=head,
            status="provisioning",
        )
        db.add(worker_group)
        db.commit()
        db.refresh(worker_group)

        # Release the pool slot before parallel Ray/dataset work (can run many minutes).
        _release_job_db(db)
        db = None

        low_memory_mode = bool(platform.non_distributed_low_memory_mode)
        launch_cdsw_workers = head and job.num_workers > 0 and not low_memory_mode
        cluster_wait_done = False

        if low_memory_mode:
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "train",
                    "Non-distributed low memory mode: skipping CDSW worker launch; "
                    "Train runs 1 rank on the session with reduced batch size.",
                ),
            )

        use_gpu = gpus_per_worker > 0
        cdsw_engines = job.num_workers if launch_cdsw_workers else 0

        if use_gpu and not launch_cdsw_workers:
            if low_memory_mode:
                detail = (
                    "GPU training needs CDSW worker engines. Turn off "
                    "'non-distributed low memory mode' in Framework settings."
                )
            elif not pool.is_remote:
                detail = (
                    "CDSW/CML workers API is not available in this process "
                    "(expected cml.workers_v1, cdsw.workers, or cdsw.launch_workers)."
                )
            elif job.num_workers < 1:
                detail = "Set Workers >= 1 for distributed GPU training."
            elif not head:
                detail = "No Ray head address on this session; check CDSW_IP_ADDRESS / framework logs."
            else:
                detail = "CDSW worker launch was skipped."
            append_job_log(db, job_id, "framework", framework_line("fail", detail))
            raise RuntimeError(detail)

        mem_err = ray_train_session_memory_error(
            cdsw_worker_engines=cdsw_engines,
            non_distributed_low_memory=low_memory_mode,
        )
        if mem_err and head:
            append_job_log(db, job_id, "framework", framework_line("fail", mem_err))
            raise RuntimeError(mem_err)

        if low_memory_mode:
            train_workers = 1
        else:
            train_workers = effective_ray_train_workers(
                max(1, job.num_workers),
                platform.ray_train_num_workers,
                cdsw_worker_engines=cdsw_engines,
            )
        train_config["ray_train_workers_scheduled"] = train_workers
        train_config["cdsw_worker_engines"] = cdsw_engines

        from app.services.job_parallel_startup import (
            parallel_shards_and_worker_provision,
            prepare_dataset_shards_parallel,
            provision_cdsw_workers_parallel,
        )

        if launch_cdsw_workers and head and train_workers > 1:
            patch, cluster_wait_done = parallel_shards_and_worker_provision(
                job_id=job_id,
                head=head,
                num_workers=job.num_workers,
                artifact_dir=artifact_dir,
                pool=pool,
                worker_spec=worker_spec,
                train_config=train_config,
                train_workers=train_workers,
                gpus_per_worker=gpus_per_worker,
                use_gpu=use_gpu,
                launch_cdsw_workers=True,
            )
            train_config.update(patch)
        elif launch_cdsw_workers and head:
            prov = provision_cdsw_workers_parallel(
                job_id=job_id,
                head=head,
                num_workers=job.num_workers,
                artifact_dir=artifact_dir,
                pool=pool,
                worker_spec=worker_spec,
                train_workers=train_workers,
                gpus_per_worker=gpus_per_worker,
                use_gpu=use_gpu,
            )
            train_config.update(prov.train_config_patch)
            cluster_wait_done = prov.cluster_wait_done
        elif train_workers > 1:
            train_config.update(
                prepare_dataset_shards_parallel(
                    job_id=job_id,
                    artifact_dir=artifact_dir,
                    train_config=train_config,
                    train_workers=train_workers,
                )
            )

        from app.training.dataset_staging import register_job_dataset_in_data_collection

        db, job = _require_job_db(db, job_id)
        project = db.get(Experiment, job.experiment_id)
        reg = register_job_dataset_in_data_collection(
            project_id=job.experiment_id,
            project_name=project.name if project else None,
            job_id=job_id,
            train_config=train_config,
        )
        if reg:
            if reg.get("access_mode") == "remote":
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "train",
                        f"Registered remote dataset under Data → {reg.get('local_path')} "
                        f"(remote: {reg.get('remote_uri')})",
                    ),
                )
            elif reg.get("reference_only"):
                shard_note = (
                    f", shards at {reg.get('cache_local_path')}"
                    if reg.get("shards_saved")
                    else ""
                )
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "train",
                        "Registered dataset under Data (reference, no copy) → "
                        f"corpus {reg.get('corpus_local_path')}{shard_note}; "
                        f"reuse root {reg.get('staging_root_ref')}; registry {reg.get('local_path')}",
                    ),
                )
            else:
                shard_note = (
                    f", {reg.get('shard_world_size')} worker shard(s) at {reg.get('cache_local_path')}"
                    if reg.get("shards_saved")
                    else ""
                )
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "train",
                        f"Registered dataset under Data → corpus {reg.get('corpus_local_path')}"
                        f"{shard_note}; collection {reg.get('local_path')}",
                    ),
                )

        from app.services.cdsw_workers import wait_for_gcs_usable
        from app.training.transformer_job import resolve_ray_address, run_transformer_training

        worker_head = head
        ray_addr = ray_driver_init_address(head) if head else resolve_ray_address(None)
        if ray_addr:
            from app.services.cai_networking import coerce_session_driver_ray_address

            ray_addr = coerce_session_driver_ray_address(ray_addr) or ray_addr
        if head and ray_addr is None:
            from app.services.cdsw_workers import ray_client_address

            ray_addr = ray_client_address(head)
        if head and worker_head and ray_addr and ray_addr != worker_head:
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "ray",
                    f"Driver Ray address {ray_addr} (workers join {worker_head})",
                ),
            )

        if not cluster_wait_done and head and ray_addr:

            def _gcs_wait_log(msg: str) -> None:
                append_job_log_isolated(job_id, "framework", framework_line("ray", msg))

            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "ray",
                    f"Waiting for Ray GCS RPC at {ray_addr} (workers join {worker_head}; "
                    "CAI: head on CDSW_IP_ADDRESS, workers dial CDSW_MASTER_IP)…",
                ),
            )
            _release_job_db(db)
            db = None
            if not wait_for_gcs_usable(ray_addr, timeout_sec=180.0, framework_log=_gcs_wait_log):
                append_job_log_isolated(
                    job_id,
                    "framework",
                    framework_line(
                        "fail",
                        f"Ray GCS not usable at {ray_addr}. On the session: "
                        "`ray stop --force`, remove data/.ray_gcs_auth_token, "
                        "`python3 scripts/start.py`. Confirm head uses pod IP "
                        "(not 127.0.0.1-only). See Cloudera distributed computing / "
                        "worker network communication docs.",
                    ),
                )
                raise RuntimeError(f"Ray GCS not available at {ray_addr}")
            db, job = _require_job_db(db, job_id)
        if ray_addr:
            os.environ["RAY_ADDRESS"] = ray_addr

        if platform.ray_local_mode and not pool.is_remote:
            ray_addr = None

        if not cluster_wait_done and launch_cdsw_workers and head:
            from app.services.ray_cluster_watch import (
                cluster_snapshot_for_wait,
                wait_for_cluster_nodes_subprocess,
            )
            from app.services.ray_driver_connect import ensure_ray_auth_token

            gcs_wait = ray_driver_init_address(head)
            min_cluster_nodes = 1 + cdsw_engines
            min_gpu = float(train_workers * max(1, gpus_per_worker)) if use_gpu else 0.0
            auth_token = ensure_ray_auth_token()
            if gcs_wait:
                _release_job_db(db)
                db = None
                wait_for_cluster_nodes_subprocess(
                    gcs_wait,
                    lambda msg: append_job_log_isolated(job_id, "framework", msg),
                    min_alive=min_cluster_nodes,
                    min_gpu=min_gpu,
                    timeout_sec=900.0 if use_gpu else 600.0,
                    auth_token=auth_token,
                )
                db, job = _require_job_db(db, job_id)
                snap = cluster_snapshot_for_wait(gcs_wait, auth_token=auth_token)
                if use_gpu and (snap is None or snap["gpu"] < min_gpu):
                    detail = (
                        f"Ray cluster reports {snap['gpu'] if snap else 0:g} GPU but "
                        f"Ray Train needs {min_gpu:g} ({train_workers} rank(s) × "
                        f"{max(1, gpus_per_worker)} GPU). "
                        "CDSW workers must join with `ray start --block` and "
                        "nvidia_gpu>0 (check worker logs for gpus=1, GPU profile, "
                        "CML_ACCELERATOR_LABEL_ID)."
                    )
                    append_job_log(db, job_id, "framework", framework_line("fail", detail))
                    raise RuntimeError(detail)

        if train_workers < job.num_workers and not low_memory_mode:
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "warn",
                    f"Capping Ray Train workers {job.num_workers} -> {train_workers}",
                ),
            )
            logger.warning(
                "Capping Ray Train workers from %s to %s (RAY_TRAIN_NUM_WORKERS / platform cap).",
                job.num_workers,
                train_workers,
            )
        elif cdsw_engines > 0:
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "train",
                    f"Distributed Train: {train_workers} Ray Train rank(s) "
                    f"across {cdsw_engines} CDSW engine(s) (world_size={train_workers}).",
                ),
            )

        _raise_if_cancelled(job_id)
        _update_job(
            db,
            job_id,
            status="running",
            ray_address=ray_addr or "local",
        )
        append_job_log(
            db,
            job_id,
            "framework",
            framework_line(
                "train",
                f"Train start driver={ray_addr or 'local'} worker_head={worker_head or 'n/a'} "
                f"workers={train_workers} gpu={use_gpu} max_steps={train_config.get('max_steps')} "
                f"epochs={job.epochs}",
            ),
        )
        worker_mem = float(getattr(pool, "memory_gb", 8.0) if launch_cdsw_workers else 0)
        if (
            low_memory_mode
            or should_apply_train_memory_savings(
                worker_memory_gb=worker_mem if cdsw_engines > 0 else None,
            )
        ) and train_config.get("per_device_train_batch_size", 4) > 2:
            train_config = {
                **train_config,
                "per_device_train_batch_size": min(
                    2, int(train_config["per_device_train_batch_size"])
                ),
            }
            logger.warning(
                "8 GiB memory profile: reduced per_device_train_batch_size to %s",
                train_config["per_device_train_batch_size"],
            )

        from app.services.ray_driver_connect import RAY_DRIVER_CONNECT_REVISION

        append_job_log(
            db,
            job_id,
            "framework",
            framework_line("train", "Ray Train job submitted to master"),
        )
        append_job_log(
            db,
            job_id,
            "framework",
            framework_line(
                "ray",
                f"Ray driver connect revision {RAY_DRIVER_CONNECT_REVISION} "
                f"(verify via GET /api/health → ray_driver_connect_revision)",
            ),
        )

        from app.services.remote_train_driver import (
            build_remote_train_driver_code,
            poll_remote_driver_result,
            remote_result_path,
            use_remote_train_driver,
            write_remote_train_payload,
        )

        remote_result_path(artifact_dir).unlink(missing_ok=True)
        use_remote = use_remote_train_driver(
            launch_cdsw_workers=launch_cdsw_workers,
            pool_is_remote=pool.is_remote,
        )

        if use_remote and ray_addr and head:
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "train",
                    "Distributed mode: launching Ray Train **driver** on a CDSW worker "
                    "engine (session keeps Ray head only; avoids co-located ray.init).",
                ),
            )
            from app.services.ray_driver_connect import ensure_ray_auth_token

            write_remote_train_payload(
                artifact_dir,
                ray_address=ray_addr,
                num_workers=train_workers,
                use_gpu=use_gpu,
                train_config=train_config,
                ray_head_address=head,
                ray_auth_token=ensure_ray_auth_token(),
            )
            driver_head = head
            driver_spec = remote_driver_launch_spec(worker_spec)
            pool.launch(
                1,
                driver_head,
                code=build_remote_train_driver_code(job_id, driver_head),
                cpu=float(driver_spec["cpu"]),
                memory_gb=float(driver_spec["memory_gb"]),
                gpus_per_worker=0,
            )
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "train",
                    "Waiting for remote driver result (see CDSW worker logs if this stalls)…",
                ),
            )

            def _poll_log(msg: str) -> None:
                append_job_log_isolated(job_id, "framework", framework_line("ray", msg))

            def _remote_poll(elapsed: float, has_result_file: bool) -> None:
                append_job_log_isolated(
                    job_id,
                    "framework",
                    framework_line(
                        "ray",
                        f"Remote driver: waiting… {int(elapsed)}s "
                        f"(result file {'present' if has_result_file else 'not yet'}) — "
                        "check CDSW driver worker logs for GCS errors",
                    ),
                )

            _release_job_db(db)
            db = None
            try:
                outcome = poll_remote_driver_result(
                    artifact_dir,
                    on_poll=_remote_poll,
                )
            except TimeoutError as exc:
                _poll_log(str(exc))
                raise
            db, job = _require_job_db(db, job_id)
            if outcome.get("status") != "succeeded":
                err = outcome.get("error") or "Remote Ray Train driver failed"
                tb = outcome.get("traceback")
                if tb:
                    append_job_log(db, job_id, "framework", framework_line("fail", err))
                    append_job_log(db, job_id, "framework", tb[-4000:])
                raise RuntimeError(err)
            result = outcome.get("result") or {"metrics": outcome.get("metrics")}
        else:
            if launch_cdsw_workers:
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "train",
                        "Ray Train driver on session pod (default). "
                        "Set PLATFORM_USE_REMOTE_TRAIN_DRIVER=1 for a CDSW driver engine.",
                    ),
                )
            _release_job_db(db)
            db = None
            result = run_transformer_training(
                ray_address=ray_addr,
                num_workers=train_workers,
                use_gpu=use_gpu,
                train_config=train_config,
            )
            db, job = _require_job_db(db, job_id)
        append_job_log(
            db,
            job_id,
            "framework",
            framework_line(
                "done",
                f"Training finished metrics={json.dumps(result.get('metrics', {}))}",
            ),
        )
        _update_job(
            db,
            job_id,
            status="succeeded",
            metrics_json=json.dumps(result.get("metrics", {})),
        )
        worker_group.status = "stopped"
        db.commit()

        if (
            cfg.task_type == "causal_lm"
            and result.get("artifact_dir")
            and Path(result["artifact_dir"]).is_dir()
        ):
            model_id = cfg.openai_model_id or default_openai_model_id(job.id, job.name)
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else None
            register_trained_model(
                db,
                job=job,
                openai_model_id=model_id,
                artifact_path=result["artifact_dir"],
                base_model=train_config["model_name"],
                task_type=cfg.task_type,
                architecture_json=cfg.architecture.model_dump_json(),
                init_mode=cfg.init_mode,
                training_metrics=metrics,
            )
    except JobCancelledError:
        db, job = _require_job_db(db, job_id)
        append_job_log(
            db,
            job_id,
            "framework",
            framework_line("info", "Training stopped by user"),
        )
        job_row = db.get(TrainingJob, job_id)
        if job_row and job_row.status != "stopped":
            _update_job(db, job_id, status="stopped", error_message="Stopped by user")
        if job_row and job_row.worker_group:
            job_row.worker_group.status = "stopped"
            db.commit()
    except Exception as exc:  # noqa: BLE001
        db, _job = _require_job_db(db, job_id)
        if is_training_job_cancel_requested(job_id):
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line("info", "Training stopped by user"),
            )
            _update_job(db, job_id, status="stopped", error_message="Stopped by user")
        else:
            logger.exception("Training job %s failed", job_id)
            append_job_log(db, job_id, "framework", framework_line("fail", f"Job failed: {exc}"))
            _update_job(db, job_id, status="failed", error_message=str(exc))
    finally:
        clear_training_job_cancel(job_id)
        _unregister_job_pool(job_id)
        if pool is not None:
            keep = os.environ.get("PLATFORM_KEEP_WORKERS_ON_FAIL", "").lower() in (
                "1",
                "true",
                "yes",
            )
            job_failed = False
            try:
                job = db.get(TrainingJob, job_id)
                job_failed = job is not None and job.status == "failed"
            except Exception:  # noqa: BLE001
                pass
            if keep and job_failed:
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "info",
                        "PLATFORM_KEEP_WORKERS_ON_FAIL=1 — CDSW worker engines left running for debugging.",
                    ),
                )
            else:
                try:
                    pool.stop_all()
                except Exception:  # noqa: BLE001
                    pass
        _release_job_db(db)


def enqueue_training_job(
    job_id: int,
    *,
    resume: bool = False,
    force_ray_head_restart: bool = False,
) -> None:
    with _job_thread_lock:
        existing = _job_threads.get(job_id)
        if existing is not None and existing.is_alive():
            raise RuntimeError(f"Training job {job_id} is already running")
        _pending_resume[job_id] = resume
        _pending_ray_force_head[job_id] = force_ray_head_restart
        thread = threading.Thread(
            target=run_training_job,
            args=(job_id,),
            daemon=True,
            name=f"training-job-{job_id}",
        )
        _job_threads[job_id] = thread
        thread.start()


def scale_job_workers(job_id: int, target: int, db: Session) -> WorkerGroup:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise ValueError("Job not found")
    if not job.worker_group:
        raise ValueError("Job has no worker group")
    from app.services.project_ray_settings import ensure_project_ray_settings

    ray_config = ensure_project_ray_settings(db, job.experiment_id)
    head = job.worker_group.head_address or resolved_ray_head(ray_config)
    if not head:
        raise ValueError("No Ray head address; run inside CML/CDSW or set RAY_HEAD_ADDRESS")

    pool_kwargs = {**worker_pool_kwargs(ray_config), "gpus_per_worker": job.gpus_per_worker}
    pool = CdswRayWorkerPool(**pool_kwargs)
    launched = pool.scale(target, head)
    job.worker_group.target_workers = target
    job.worker_group.worker_ids_json = worker_ids_to_json(launched.worker_ids)
    job.num_workers = target
    db.commit()
    db.refresh(job.worker_group)
    return job.worker_group
