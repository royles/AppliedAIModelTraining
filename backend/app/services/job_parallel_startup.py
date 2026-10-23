"""Overlap I/O-bound dataset prep with Ray head / worker provisioning."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from app.database import SessionLocal
from app.models import TrainingJob, WorkerGroup
from app.services.job_log import append_job_log
from app.services.log_flair import framework_line, task_done_line, task_start_line
from app.services.download_progress import DownloadProgressState
from app.services.task_heartbeat import provisioning_heartbeat
from app.training.from_scratch import prepare_from_scratch_training
from app.training.transformer_config import TransformerTrainingConfig

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def _job_framework_log(job_id: int) -> LogFn:
    def log(msg: str) -> None:
        db = SessionLocal()
        try:
            append_job_log(db, job_id, "framework", msg)
        finally:
            db.close()

    return log


def _log_phase_start(log: LogFn, kind: str, label: str) -> float:
    line = task_start_line(kind, label)
    log(line)
    logger.info("%s", label)
    return time.monotonic()


def _log_phase_done(
    log: LogFn,
    t0: float,
    label: str,
    *,
    detail: str = "",
) -> None:
    elapsed = time.monotonic() - t0
    line = task_done_line(label, seconds=elapsed, detail=detail)
    log(line)
    logger.info("%s — %s (%.0fs)", label, detail or "done", elapsed)


def prepare_train_config_dataset(
    cfg: TransformerTrainingConfig,
    train_config: dict[str, Any],
    *,
    job_id: int,
    resume: bool,
) -> dict[str, Any]:
    """Prompt-format detection and from-scratch corpus load (I/O bound)."""
    log = _job_framework_log(job_id)
    tc = dict(train_config)

    if cfg.task_type == "causal_lm":
        from app.training.dataset_sharding import load_training_corpus_staged
        from app.training.dataset_staging import shared_dataset_dirs_from_config
        from app.training.lm_text_utils import resolve_prompt_format

        setting = tc.get("prompt_format") or cfg.prompt_format or "auto"
        if setting == "auto":
            artifact_dir = str(tc.get("artifact_dir") or "")
            shared_dirs = shared_dataset_dirs_from_config(tc)
            t0 = _log_phase_start(log, "train", "Dataset load (corpus)")
            dl_progress = DownloadProgressState()
            try:
                with provisioning_heartbeat(
                    log, "Dataset load", download_progress=dl_progress
                ):
                    sample_texts, _ = load_training_corpus_staged(
                        artifact_dir=artifact_dir or None,
                        dataset_cfg=cfg.dataset.model_dump(),
                        task_type=cfg.task_type,
                        shared_dirs=shared_dirs,
                        download_progress=dl_progress,
                    )
                tc["prompt_format"] = resolve_prompt_format("auto", sample_texts)
                _log_phase_done(
                    log,
                    t0,
                    "Dataset load",
                    detail=f"{len(sample_texts)} samples",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not detect prompt_format: %s", exc)
                tc["prompt_format"] = "raw_lm"
                _log_phase_done(log, t0, "Dataset load", detail="failed, using raw_lm")
        else:
            tc["prompt_format"] = resolve_prompt_format(setting, None)

    if cfg.init_mode == "from_scratch":
        t0 = _log_phase_start(log, "train", "From-scratch tokenizer & corpus")
        with provisioning_heartbeat(log, "From-scratch prep"):
            tc = prepare_from_scratch_training(
                cfg,
                tc,
                log=log,
                skip_tokenizer_if_present=resume,
            )
        _log_phase_done(log, t0, "From-scratch prep")

    return tc


def ensure_ray_head_parallel(
    *,
    job_id: int,
    needs_cai_ray_head: bool,
    force_restart: bool,
    head_before: str | None,
) -> str | None:
    if not needs_cai_ray_head:
        return head_before

    log = _job_framework_log(job_id)
    from app.services.ray_cai_networking import ensure_ray_head_for_workers
    from app.services.ray_gcs_auth import bind_gcs_auth_for_address

    t0 = _log_phase_start(
        log,
        "ray",
        "Ray cluster setup" + (" (full restart)" if force_restart else ""),
    )

    def _ray_head_log(msg: str) -> None:
        line = msg if msg.startswith("[") else framework_line("ray", msg)
        log(line)
        logger.info("%s", line)

    with provisioning_heartbeat(log, "Ray cluster setup"):
        head = ensure_ray_head_for_workers(
            framework_log=_ray_head_log,
            force_restart=force_restart,
        )
    bind_gcs_auth_for_address(head, required=True)
    _log_phase_done(log, t0, "Ray cluster setup", detail=head or "")
    return head


def parallel_dataset_prep_and_ray_head(
    *,
    job_id: int,
    cfg: TransformerTrainingConfig,
    train_config: dict[str, Any],
    needs_cai_ray_head: bool,
    force_ray_head_restart: bool,
    head_before: str | None,
    resume: bool,
    skip_ray_head_setup: bool = False,
) -> tuple[dict[str, Any], str | None]:
    log = _job_framework_log(job_id)
    log(framework_line("info", "Parallel: dataset prep + Ray head"))
    with provisioning_heartbeat(log, "Dataset prep + Ray head"):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="job-start") as pool:
            f_data = pool.submit(
                prepare_train_config_dataset,
                cfg,
                train_config,
                job_id=job_id,
                resume=resume,
            )
            if skip_ray_head_setup and head_before and needs_cai_ray_head:
                log(
                    framework_line(
                        "ray",
                        f"Ray head at {head_before} (full restart done — skipping parallel re-check)",
                    )
                )

                def _return_head() -> str | None:
                    return head_before

                f_ray = pool.submit(_return_head)
            else:
                f_ray = pool.submit(
                    ensure_ray_head_parallel,
                    job_id=job_id,
                    needs_cai_ray_head=needs_cai_ray_head,
                    force_restart=force_ray_head_restart,
                    head_before=head_before,
                )
            tc = f_data.result()
            head = f_ray.result()
    log(framework_line("done", "Parallel: dataset prep + Ray head"))
    return tc, head


def prepare_dataset_shards_parallel(
    *,
    job_id: int,
    artifact_dir: str,
    train_config: dict[str, Any],
    train_workers: int,
) -> dict[str, Any]:
    log = _job_framework_log(job_id)
    patch: dict[str, Any] = {"dataset_shards_prepared": False}
    if train_workers <= 1:
        return patch

    from app.training.dataset_sharding import (
        prepare_dataset_shards_on_driver,
        prepared_dataset_shards_meta,
    )
    from app.training.dataset_staging import shared_dataset_dirs_from_config

    shared_dirs = shared_dataset_dirs_from_config(train_config)
    dataset_cfg = train_config.get("dataset") or {
        "source": "builtin",
        "builtin_name": "platform_demo",
    }
    task_type = str(train_config.get("task_type") or "causal_lm")
    existing = prepared_dataset_shards_meta(
        artifact_dir=artifact_dir,
        dataset_cfg=dataset_cfg,
        task_type=task_type,
        world_size=train_workers,
        shared_dirs=shared_dirs,
    )
    if existing is not None:
        root = existing.get("staging_root") or existing.get("cache_dir") or artifact_dir
        log(
            framework_line(
                "train",
                f"Reusing {train_workers} dataset shard(s) on disk "
                f"({existing.get('total_examples')} examples) at {root}",
            )
        )
        patch["dataset_shards_prepared"] = True
        return patch

    t0 = _log_phase_start(log, "train", "Dataset download & sharding")
    dl_progress = DownloadProgressState()
    with provisioning_heartbeat(
        log, "Dataset download & sharding", download_progress=dl_progress
    ):
        meta = prepare_dataset_shards_on_driver(
            artifact_dir=artifact_dir,
            dataset_cfg=dataset_cfg,
            task_type=task_type,
            world_size=train_workers,
            shared_dirs=shared_dirs,
            download_progress=dl_progress,
        )
    patch["dataset_shards_prepared"] = True
    _log_phase_done(
        log,
        t0,
        "Dataset sharding",
        detail=f"{meta.get('total_examples')} ex → {train_workers} shards",
    )
    return patch


@dataclass
class CdswProvisionResult:
    train_config_patch: dict[str, Any] = field(default_factory=dict)
    cluster_wait_done: bool = False


def provision_cdsw_workers_parallel(
    *,
    job_id: int,
    head: str,
    num_workers: int,
    artifact_dir: str,
    pool: Any,
    worker_spec: dict[str, Any],
    train_workers: int,
    gpus_per_worker: int,
    use_gpu: bool,
) -> CdswProvisionResult:
    """Probe, launch CDSW engines, wait for GCS and cluster nodes."""
    log = _job_framework_log(job_id)
    patch: dict[str, Any] = {
        "expected_cdsw_worker_nodes": num_workers,
        "ray_head_address": head,
    }
    result = CdswProvisionResult(train_config_patch=patch)

    from app.services.cai_networking import resolve_worker_ray_head_address_with_source
    from app.services.cdsw_workers import (
        ray_driver_init_address,
        wait_for_gcs_usable,
        worker_ids_to_json,
    )
    from app.services.ray_cai_networking import (
        WORKER_CONNECTIVITY_PROBE_REVISION,
        build_worker_connectivity_probe_code,
        describe_worker_probe_failure,
        poll_worker_connectivity_probe,
        worker_connectivity_probe_passed,
    )
    from app.services.ray_cluster_watch import (
        cluster_snapshot_for_wait,
        log_cdsw_worker_api_status,
        log_cdsw_worker_ids,
        wait_for_cluster_nodes_subprocess,
    )
    from app.services.ray_driver_connect import ensure_ray_auth_token
    from app.services.ray_gcs_auth import project_gcs_auth_token_path, require_ray_gcs_auth_token
    from app.services.worker_resources import format_worker_spec_line
    from pathlib import Path

    log(
        framework_line(
            "worker",
            format_worker_spec_line(worker_spec, num_workers=num_workers),
        )
    )

    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job or not job.worker_group:
            raise RuntimeError("Worker group missing during CDSW provisioning")
        job.worker_group.head_address = head
        db.commit()
    finally:
        db.close()

    worker_dial, worker_dial_via = resolve_worker_ray_head_address_with_source(head)
    log(
        framework_line(
            "worker",
            f"Launching {num_workers} CDSW GPU engine(s); workers dial master GCS "
            f"{worker_dial} (Cloudera CDSW_MASTER_IP→master, session via {worker_dial_via})",
        )
    )
    probe_path = str(Path(artifact_dir) / "ray_worker_probe.json")
    Path(probe_path).unlink(missing_ok=True)
    log(
        framework_line(
            "ray",
            "Running CDSW worker→master connectivity probe (TCP + ray.init)…",
        )
    )
    probe_token = require_ray_gcs_auth_token(head)

    def _gcs_log(msg: str) -> None:
        log(framework_line("ray", msg))

    if not wait_for_gcs_usable(head, timeout_sec=180.0, framework_log=_gcs_log):
        msg = (
            f"Ray GCS not usable at {head} before worker probe. "
            "Restart head on master pod IP (ray stop --force; start.py). "
            "See Cloudera ML distributed computing — Worker Network Communication."
        )
        log(framework_line("fail", msg))
        raise RuntimeError(msg)

    log(
        framework_line(
            "ray",
            f"Worker probe revision {WORKER_CONNECTIVITY_PROBE_REVISION}; "
            f"GCS auth token synced ({project_gcs_auth_token_path().name})",
        )
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
    if not worker_connectivity_probe_passed(probe):
        detail = describe_worker_probe_failure(probe, head)
        log(framework_line("fail", detail))
        raise RuntimeError(detail)
    log(
        framework_line(
            "ray",
            f"Worker connectivity OK: master GCS {probe.get('head')} via "
            f"{probe.get('head_via', '?')} (nodes={probe.get('nodes', '?')}); launching GPU workers",
        )
    )
    launched = pool.launch(num_workers, head)
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job or not job.worker_group:
            raise RuntimeError("Worker group missing after CDSW launch")
        worker_group = job.worker_group
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
    finally:
        db.close()

    from app.services.cai_networking import coerce_session_driver_ray_address
    from app.services.cdsw_workers import ray_client_address

    worker_head = head
    ray_addr = ray_driver_init_address(head) if head else None
    if ray_addr:
        ray_addr = coerce_session_driver_ray_address(ray_addr) or ray_addr
    if head and ray_addr is None:
        ray_addr = ray_client_address(head)
    if head and worker_head and ray_addr and ray_addr != worker_head:
        log(
            framework_line(
                "ray",
                f"Driver Ray address {ray_addr} (workers join {worker_head})",
            )
        )

    if head and ray_addr:

        def _cluster_gcs_log(msg: str) -> None:
            log(framework_line("ray", msg))

        t_gcs = _log_phase_start(log, "ray", "Ray GCS RPC wait")
        if not wait_for_gcs_usable(ray_addr, timeout_sec=180.0, framework_log=_cluster_gcs_log):
            msg = (
                f"Ray GCS not usable at {ray_addr}. "
                "Workers must reach master via CDSW_MASTER_IP (Cloudera worker networking)."
            )
            log(framework_line("fail", msg))
            raise RuntimeError(f"Ray GCS not available at {ray_addr}")
        _log_phase_done(log, t_gcs, "Ray GCS RPC wait", detail=ray_addr)
        result.cluster_wait_done = True

        gcs_wait = ray_driver_init_address(head)
        min_cluster_nodes = 1 + num_workers
        min_gpu = float(train_workers * max(1, gpus_per_worker)) if use_gpu else 0.0
        auth_token = ensure_ray_auth_token()
        if gcs_wait:
            gpu_note = f", need {min_gpu:g} GPU" if use_gpu else ""
            t_nodes = _log_phase_start(
                log,
                "ray",
                f"Ray worker join (≥{min_cluster_nodes} nodes{gpu_note})",
            )
            final_n = wait_for_cluster_nodes_subprocess(
                gcs_wait,
                log,
                min_alive=min_cluster_nodes,
                min_gpu=min_gpu,
                timeout_sec=900.0 if use_gpu else 600.0,
                auth_token=auth_token,
            )
            snap = cluster_snapshot_for_wait(gcs_wait, auth_token=auth_token)
            node_detail = f"{final_n} node(s)"
            if snap:
                node_detail += f", {snap.get('gpu', 0):g} GPU"
            _log_phase_done(log, t_nodes, "Ray worker join", detail=node_detail)
            if use_gpu and (snap is None or snap["gpu"] < min_gpu):
                detail = (
                    f"Ray cluster reports {snap['gpu'] if snap else 0:g} GPU but "
                    f"Ray Train needs {min_gpu:g} ({train_workers} rank(s) × "
                    f"{max(1, gpus_per_worker)} GPU). "
                    "CDSW workers must join with `ray start --block` and "
                    "nvidia_gpu>0 (check worker logs for gpus=1, GPU profile, "
                    "CML_ACCELERATOR_LABEL_ID)."
                )
                log(framework_line("fail", detail))
                raise RuntimeError(detail)

    log(framework_line("done", "CDSW workers provisioned"))
    return result


def parallel_shards_and_worker_provision(
    *,
    job_id: int,
    head: str,
    num_workers: int,
    artifact_dir: str,
    pool: Any,
    worker_spec: dict[str, Any],
    train_config: dict[str, Any],
    train_workers: int,
    gpus_per_worker: int,
    use_gpu: bool,
    launch_cdsw_workers: bool,
) -> tuple[dict[str, Any], bool]:
    """
    While CDSW workers join the cluster, load the corpus and write Train shards on the driver.
    Returns (train_config patch, cluster_wait_already_done).
    """
    log = _job_framework_log(job_id)
    log(framework_line("info", "Parallel: dataset sharding + worker provisioning"))
    shard_patch: dict[str, Any] = {}
    provision = CdswProvisionResult()

    with provisioning_heartbeat(log, "Dataset sharding + worker provisioning"):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="job-provision") as executor:
            f_shards = executor.submit(
                prepare_dataset_shards_parallel,
                job_id=job_id,
                artifact_dir=artifact_dir,
                train_config=train_config,
                train_workers=train_workers,
            )
            if launch_cdsw_workers:
                f_prov = executor.submit(
                    provision_cdsw_workers_parallel,
                    job_id=job_id,
                    head=head,
                    num_workers=num_workers,
                    artifact_dir=artifact_dir,
                    pool=pool,
                    worker_spec=worker_spec,
                    train_workers=train_workers,
                    gpus_per_worker=gpus_per_worker,
                    use_gpu=use_gpu,
                )
                wait([f_shards, f_prov], return_when=ALL_COMPLETED)
                shard_patch = f_shards.result()
                provision = f_prov.result()
            else:
                shard_patch = f_shards.result()

    log(framework_line("done", "Parallel: sharding + worker provisioning"))
    return {**provision.train_config_patch, **shard_patch}, provision.cluster_wait_done
