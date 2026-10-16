from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.resource_hints import effective_ray_train_workers, is_low_memory_container
from app.database import SessionLocal
from app.models import TrainingJob, WorkerGroup
from app.services.cdsw_workers import (
    CdswRayWorkerPool,
    worker_ids_to_json,
)
from app.services.platform_settings import (
    ensure_platform_settings,
    resolved_ray_head,
    worker_pool_kwargs,
)
from app.services.job_log import append_job_log
from app.services.model_registry import default_openai_model_id, register_trained_model
from app.training.from_scratch import prepare_from_scratch_training
from app.training.transformer_config import TransformerTrainingConfig

logger = logging.getLogger(__name__)


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
        append_job_log(db, job_id, "framework", "Job queued; starting provisioning.")

        cfg = resolve_job_config(job)
        artifact_dir = str((REPO_ROOT / "data" / "models" / f"job_{job_id}").resolve())
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        train_config = cfg.to_train_dict(job_id=job_id, artifact_dir=artifact_dir)

        def framework_log(msg: str) -> None:
            append_job_log(db, job_id, "framework", msg)

        if cfg.init_mode == "from_scratch":
            try:
                train_config = prepare_from_scratch_training(
                    cfg,
                    train_config,
                    log=framework_log,
                )
            except ValueError as exc:
                append_job_log(db, job_id, "framework", f"From-scratch prep failed: {exc}")
                raise

        platform = ensure_platform_settings(db)
        head = resolved_ray_head(platform)
        pool_kwargs = {**worker_pool_kwargs(platform), "gpus_per_worker": job.gpus_per_worker}
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

        if head and job.num_workers > 0:
            append_job_log(
                db,
                job_id,
                "framework",
                f"Launching {job.num_workers} CDSW worker(s) for Ray head {head}.",
            )
            launched = pool.launch(job.num_workers, head)
            worker_group.worker_ids_json = worker_ids_to_json(launched.worker_ids)
            worker_group.status = "running"
            db.commit()
            append_job_log(
                db,
                job_id,
                "framework",
                f"Workers launched: {launched.worker_ids}",
            )

        from app.training.transformer_job import resolve_ray_address, run_transformer_training

        ray_addr = resolve_ray_address(head)
        if head and ray_addr is None:
            from app.services.cdsw_workers import ray_client_address

            ray_addr = ray_client_address(head)

        use_gpu = job.gpus_per_worker > 0
        if platform.ray_local_mode and not pool.is_remote:
            ray_addr = None

        train_workers = effective_ray_train_workers(
            max(1, job.num_workers),
            platform.ray_train_num_workers,
        )
        if train_workers < job.num_workers:
            append_job_log(
                db,
                job_id,
                "framework",
                f"Capping Ray Train workers from {job.num_workers} to {train_workers}.",
            )
            logger.warning(
                "Capping Ray Train workers from %s to %s (small container / "
                "PLATFORM_CAI_LOW_MEMORY / RAY_TRAIN_NUM_WORKERS).",
                job.num_workers,
                train_workers,
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
            f"Training starting (Ray={ray_addr or 'local'}, train_workers={train_workers}, gpu={use_gpu}).",
        )
        if is_low_memory_container() and train_config.get("per_device_train_batch_size", 4) > 2:
            train_config = {
                **train_config,
                "per_device_train_batch_size": min(
                    2, int(train_config["per_device_train_batch_size"])
                ),
            }
            logger.warning(
                "Low-memory container: reduced per_device_train_batch_size to %s",
                train_config["per_device_train_batch_size"],
            )

        append_job_log(db, job_id, "framework", "Ray Train job submitted.")
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
            f"Training finished. Metrics: {json.dumps(result.get('metrics', {}))}",
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
            register_trained_model(
                db,
                job_id=job.id,
                openai_model_id=model_id,
                artifact_path=result["artifact_dir"],
                base_model=train_config["model_name"],
                task_type=cfg.task_type,
                architecture_json=cfg.architecture.model_dump_json(),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training job %s failed", job_id)
        append_job_log(db, job_id, "framework", f"Job failed: {exc}")
        _update_job(db, job_id, status="failed", error_message=str(exc))
    finally:
        if pool is not None:
            try:
                pool.stop_all()
            except Exception:  # noqa: BLE001
                pass
        db.close()


def enqueue_training_job(job_id: int) -> None:
    thread = threading.Thread(target=run_training_job, args=(job_id,), daemon=True)
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
