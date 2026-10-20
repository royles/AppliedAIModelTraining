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
from app.models import TrainingJob, WorkerGroup
from app.services.cdsw_workers import (
    CdswRayWorkerPool,
    ray_driver_init_address,
    worker_ids_to_json,
)
from app.services.platform_settings import (
    ensure_platform_settings,
    resolved_ray_head,
    worker_pool_kwargs,
)
from app.services.job_log import append_job_log
from app.services.log_flair import framework_banner, framework_line
from app.services.ray_cluster_watch import (
    log_cdsw_worker_api_status,
    log_cdsw_worker_ids,
)
from app.services.model_registry import default_openai_model_id, register_trained_model
from app.training.from_scratch import prepare_from_scratch_training
from app.training.transformer_config import TransformerTrainingConfig

logger = logging.getLogger(__name__)

_job_threads: dict[int, threading.Thread] = {}
_job_thread_lock = threading.Lock()
_pending_resume: dict[int, bool] = {}


def job_is_running(job_id: int) -> bool:
    with _job_thread_lock:
        thread = _job_threads.get(job_id)
        return thread is not None and thread.is_alive()


def _pop_resume_flag(job_id: int) -> bool:
    with _job_thread_lock:
        return _pending_resume.pop(job_id, False)


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


def run_training_job(job_id: int) -> None:
    db = SessionLocal()
    pool: CdswRayWorkerPool | None = None
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
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
        train_config = apply_resume_to_train_config(
            train_config, artifact_dir, resume=resume_run
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

        def framework_log(msg: str) -> None:
            append_job_log(db, job_id, "framework", msg)

        if cfg.init_mode == "from_scratch":
            try:
                train_config = prepare_from_scratch_training(
                    cfg,
                    train_config,
                    log=framework_log,
                    skip_tokenizer_if_present=bool(train_config.get("resume")),
                )
            except ValueError as exc:
                append_job_log(db, job_id, "framework", f"From-scratch prep failed: {exc}")
                raise

        platform = ensure_platform_settings(db)
        head = resolved_ray_head(platform)
        from app.services.cai_networking import format_ray_head, master_engine_ip, parse_gcs_host_port

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
        pool_kwargs = {**worker_pool_kwargs(platform), "gpus_per_worker": gpus_per_worker}
        pool = CdswRayWorkerPool(**pool_kwargs)

        worker_group = WorkerGroup(
            job_id=job.id,
            target_workers=job.num_workers,
            head_address=head,
            status="provisioning",
        )
        db.add(worker_group)
        db.commit()
        db.refresh(worker_group)

        low_memory_mode = bool(platform.non_distributed_low_memory_mode)
        launch_cdsw_workers = head and job.num_workers > 0 and not low_memory_mode

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

        if launch_cdsw_workers:
            from app.services.ray_cai_networking import (
                build_worker_connectivity_probe_code,
                ensure_ray_head_for_workers,
                poll_worker_connectivity_probe,
            )

            def _fw(msg: str) -> None:
                append_job_log(db, job_id, "framework", msg)

            from app.services.ray_gcs_auth import bind_gcs_auth_for_address

            head = ensure_ray_head_for_workers(
                framework_log=lambda m: _fw(framework_line("ray", m)),
                force_restart=False,
            )
            bind_gcs_auth_for_address(head, required=True)
            worker_group.head_address = head
            db.commit()
            from app.services.cai_networking import resolve_worker_ray_head_address_with_source

            worker_dial, worker_dial_via = resolve_worker_ray_head_address_with_source(head)
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "worker",
                    f"Launching {job.num_workers} CDSW GPU engine(s); workers dial master GCS "
                    f"{worker_dial} (Cloudera CDSW_MASTER_IP→master, session via {worker_dial_via})",
                ),
            )
            probe_path = str(Path(artifact_dir) / "ray_worker_probe.json")
            Path(probe_path).unlink(missing_ok=True)
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "ray",
                    "Running CDSW worker→master connectivity probe (TCP + ray.init)…",
                ),
            )
            from app.services.ray_cai_networking import WORKER_CONNECTIVITY_PROBE_REVISION
            from app.services.ray_gcs_auth import project_gcs_auth_token_path, require_ray_gcs_auth_token

            probe_token = require_ray_gcs_auth_token(head)
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "ray",
                    f"Worker probe revision {WORKER_CONNECTIVITY_PROBE_REVISION}; "
                    f"GCS auth token synced ({project_gcs_auth_token_path().name})",
                ),
            )
            pool.launch(
                1,
                head,
                code=build_worker_connectivity_probe_code(
                    head, probe_path, ray_auth_token=probe_token
                ),
                cpu=1.0,
                memory_gb=4.0,
                gpus_per_worker=0,
            )
            probe = poll_worker_connectivity_probe(probe_path, timeout_sec=300.0)
            from app.services.ray_cai_networking import (
                describe_worker_probe_failure,
                worker_connectivity_probe_passed,
            )

            if not worker_connectivity_probe_passed(probe):
                detail = describe_worker_probe_failure(probe, head)
                append_job_log(db, job_id, "framework", framework_line("fail", detail))
                raise RuntimeError(detail)
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "ray",
                    f"Worker connectivity OK: master GCS {probe.get('head')} via "
                    f"{probe.get('head_via', '?')} (nodes={probe.get('nodes', '?')}); launching GPU workers",
                ),
            )
            launched = pool.launch(job.num_workers, head)
            worker_group.worker_ids_json = worker_ids_to_json(launched.worker_ids)
            worker_group.status = "running"
            db.commit()
            log_cdsw_worker_ids(db, job_id, launched.worker_ids, head)
            log_cdsw_worker_api_status(
                db,
                job_id,
                pool.list_workers_safe if pool.is_remote else None,
                launched.worker_ids,
            )
            train_config["expected_cdsw_worker_nodes"] = job.num_workers
            train_config["ray_head_address"] = head

        from app.services.cdsw_workers import wait_for_gcs_port
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

        if head and ray_addr:
            gcs_host, _, port_s = ray_addr.partition(":")
            port = int(port_s or "6379")
            append_job_log(
                db,
                job_id,
                "framework",
                framework_line(
                    "ray",
                    f"Waiting for Ray GCS on {ray_addr} (workers join {worker_head})…",
                ),
            )
            if not wait_for_gcs_port(gcs_host, port, timeout_sec=180.0):
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "fail",
                        f"Ray GCS not reachable at {ray_addr} — run scripts/start.py "
                        "or check start-user.log for ray start --head",
                    ),
                )
                raise RuntimeError(f"Ray GCS not available at {ray_addr}")
        if ray_addr:
            os.environ["RAY_ADDRESS"] = ray_addr

        use_gpu = gpus_per_worker > 0
        if platform.ray_local_mode and not pool.is_remote:
            ray_addr = None

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
                detail = "No Ray head address; run scripts/start.py on the master session."
            else:
                detail = "CDSW worker launch was skipped."
            append_job_log(db, job_id, "framework", framework_line("fail", detail))
            raise RuntimeError(detail)

        mem_err = ray_train_session_memory_error(
            cdsw_worker_engines=cdsw_engines,
            non_distributed_low_memory=low_memory_mode,
        )
        if mem_err and ray_addr:
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

        if launch_cdsw_workers and head:
            from app.services.ray_cluster_watch import (
                ray_cluster_snapshot_subprocess,
                wait_for_cluster_nodes_subprocess,
            )
            from app.services.ray_driver_connect import ensure_ray_auth_token

            gcs_wait = ray_driver_init_address(head)
            min_cluster_nodes = 1 + cdsw_engines
            min_gpu = float(train_workers * max(1, gpus_per_worker)) if use_gpu else 0.0
            auth_token = ensure_ray_auth_token()
            if gcs_wait:
                wait_for_cluster_nodes_subprocess(
                    gcs_wait,
                    lambda msg: append_job_log(db, job_id, "framework", msg),
                    min_alive=min_cluster_nodes,
                    min_gpu=min_gpu,
                    timeout_sec=900.0 if use_gpu else 600.0,
                    auth_token=auth_token,
                )
                snap = ray_cluster_snapshot_subprocess(gcs_wait, auth_token=auth_token)
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
            pool.launch(
                1,
                driver_head,
                code=build_remote_train_driver_code(job_id, driver_head),
                cpu=2.0,
                memory_gb=12.0,
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
                append_job_log(db, job_id, "framework", framework_line("ray", msg))

            def _remote_poll(elapsed: float, has_result_file: bool) -> None:
                append_job_log(
                    db,
                    job_id,
                    "framework",
                    framework_line(
                        "ray",
                        f"Remote driver: waiting… {int(elapsed)}s "
                        f"(result file {'present' if has_result_file else 'not yet'}) — "
                        "check CDSW driver worker logs for GCS errors",
                    ),
                )

            try:
                outcome = poll_remote_driver_result(
                    artifact_dir,
                    on_poll=_remote_poll,
                )
            except TimeoutError as exc:
                _poll_log(str(exc))
                raise
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
            result = run_transformer_training(
                ray_address=ray_addr,
                num_workers=train_workers,
                use_gpu=use_gpu,
                train_config=train_config,
            )
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training job %s failed", job_id)
        append_job_log(db, job_id, "framework", framework_line("fail", f"Job failed: {exc}"))
        _update_job(db, job_id, status="failed", error_message=str(exc))
    finally:
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
        db.close()


def enqueue_training_job(job_id: int, *, resume: bool = False) -> None:
    with _job_thread_lock:
        existing = _job_threads.get(job_id)
        if existing is not None and existing.is_alive():
            raise RuntimeError(f"Training job {job_id} is already running")
        _pending_resume[job_id] = resume
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
    platform = ensure_platform_settings(db)
    head = job.worker_group.head_address or resolved_ray_head(platform)
    if not head:
        raise ValueError("No Ray head address; run inside CML/CDSW or set RAY_HEAD_ADDRESS")

    pool_kwargs = {**worker_pool_kwargs(platform), "gpus_per_worker": job.gpus_per_worker}
    pool = CdswRayWorkerPool(**pool_kwargs)
    launched = pool.scale(target, head)
    job.worker_group.target_workers = target
    job.worker_group.worker_ids_json = worker_ids_to_json(launched.worker_ids)
    job.num_workers = target
    db.commit()
    db.refresh(job.worker_group)
    return job.worker_group
