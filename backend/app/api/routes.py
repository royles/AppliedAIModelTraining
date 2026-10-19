import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Experiment, TrainingJob
from app.schemas import (
    ArchitectureSummary,
    ExperimentCreate,
    ExperimentRead,
    HealthResponse,
    ScaleWorkersRequest,
    TrainingJobCreate,
    TrainingJobRead,
    JobLogsRead,
    TransformerPresetRead,
    WorkerGroupRead,
    PlatformSettingsRead,
    PlatformSettingsUpdate,
)
from app.resource_hints import memory_guidance_for_health
from app.services import cdsw_workers
from app.services.job_delete import delete_training_job
from app.services.job_log import build_job_logs
from app.services.job_runner import enqueue_training_job, scale_job_workers
from app.services.model_registry import get_trained_model_for_job, openai_model_ids_for_jobs
from app.services.platform_settings import get_platform_settings, update_platform_settings
from app.training.architecture import TransformerArchitectureConfig
from app.training.dataset_config import TrainingDatasetConfig
from app.training.dataset_loader import BUILTIN_DATASET_CATALOG, list_dataset_files, load_training_samples
from app.training.from_scratch import from_scratch_resource_guidance
from app.static_ui import read_ui_build_info
from app.training.transformer_config import (
    TransformerTrainingConfig,
    list_presets,
)

router = APIRouter()


def _job_to_read(db: Session, job: TrainingJob) -> TrainingJobRead:
    base = TrainingJobRead.model_validate(job)
    record = get_trained_model_for_job(db, job.id)
    if record:
        return base.model_copy(update={"openai_model_id": record.openai_model_id})
    return base


def _jobs_to_read(db: Session, jobs: list[TrainingJob]) -> list[TrainingJobRead]:
    model_ids = openai_model_ids_for_jobs(db, [j.id for j in jobs])
    out: list[TrainingJobRead] = []
    for job in jobs:
        base = TrainingJobRead.model_validate(job)
        mid = model_ids.get(job.id)
        out.append(base.model_copy(update={"openai_model_id": mid}) if mid else base)
    return out


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key == "architecture"
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _build_config(payload: TrainingJobCreate) -> TransformerTrainingConfig:
    overrides: dict[str, Any] = {}
    if payload.transformer:
        overrides = payload.transformer.model_dump(exclude_unset=True, exclude_none=True)
    if payload.model_name:
        overrides["base_model"] = payload.model_name
        overrides.setdefault("init_mode", "pretrained")
    if payload.batch_size is not None:
        overrides["per_device_train_batch_size"] = payload.batch_size
    if payload.learning_rate is not None:
        overrides["learning_rate"] = payload.learning_rate
    if payload.max_steps is not None:
        overrides["max_steps"] = payload.max_steps
    if payload.openai_model_id:
        overrides["openai_model_id"] = payload.openai_model_id
    base = TransformerTrainingConfig.merge_preset(payload.preset, None)
    if not overrides:
        return base
    merged = _deep_merge_dict(base.model_dump(), overrides)
    return TransformerTrainingConfig.model_validate(merged)


@router.get("/ui/build-info")
def ui_build_info() -> dict[str, str]:
    """Which committed frontend/dist bundle is on disk (helps debug stale CAI UX)."""
    return read_ui_build_info()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import ray

    runtime = "initialized" if ray.is_initialized() else "not_initialized"
    return HealthResponse(
        status="ok",
        ray_runtime=runtime,
        cdsw_available=cdsw_workers.cdsw_available(),
        details={
            "ray_head": cdsw_workers.ray_head_address(),
            "docs": "https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html",
            "memory": memory_guidance_for_health(),
        },
    )


@router.get("/platform/settings", response_model=PlatformSettingsRead)
def read_platform_settings(db: Session = Depends(get_db)) -> PlatformSettingsRead:
    return get_platform_settings(db)


@router.put("/platform/settings", response_model=PlatformSettingsRead)
def write_platform_settings(
    payload: PlatformSettingsUpdate, db: Session = Depends(get_db)
) -> PlatformSettingsRead:
    return update_platform_settings(db, payload)


@router.post("/training/from-scratch/guidance")
def from_scratch_guidance(payload: TransformerTrainingConfig) -> dict:
    return from_scratch_resource_guidance(payload)


@router.post("/training/dataset/preview")
def dataset_preview(
    dataset: TrainingDatasetConfig,
    task_type: str = "causal_lm",
) -> dict[str, int | str]:
    try:
        texts, _ = load_training_samples(dataset.model_dump(), task_type=task_type)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Hugging Face datasets not available: {exc}",
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    total_chars = sum(len(t) for t in texts)
    return {
        "examples": len(texts),
        "total_characters": total_chars,
        "source": dataset.source,
    }


@router.get("/training/datasets/builtin")
def builtin_datasets() -> dict[str, dict[str, str]]:
    return BUILTIN_DATASET_CATALOG


@router.get("/training/datasets/files")
def dataset_files() -> list[dict[str, str]]:
    return list_dataset_files()


@router.get("/transformer/presets", response_model=list[TransformerPresetRead])
def transformer_presets() -> list[TransformerPresetRead]:
    return [
        TransformerPresetRead(
            name=p.name,
            title=p.title,
            description=p.description,
            config=p.config,
        )
        for p in list_presets()
    ]


@router.post("/transformer/architecture/estimate", response_model=ArchitectureSummary)
def estimate_architecture(architecture: TransformerArchitectureConfig) -> ArchitectureSummary:
    return ArchitectureSummary(
        architecture=architecture,
        estimated_parameters=architecture.parameter_count_estimate(),
    )


def _list_projects(db: Session) -> list[Experiment]:
    return list(db.scalars(select(Experiment).order_by(Experiment.created_at.desc())))


@router.get("/experiments", response_model=list[ExperimentRead], include_in_schema=False)
@router.get("/projects", response_model=list[ExperimentRead])
def list_projects(db: Session = Depends(get_db)) -> list[Experiment]:
    return _list_projects(db)


@router.post("/experiments", response_model=ExperimentRead, include_in_schema=False)
@router.post("/projects", response_model=ExperimentRead)
def create_project(payload: ExperimentCreate, db: Session = Depends(get_db)) -> Experiment:
    exp = Experiment(name=payload.name, description=payload.description)
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/jobs", response_model=list[TrainingJobRead])
def list_jobs(db: Session = Depends(get_db)) -> list[TrainingJobRead]:
    jobs = list(db.scalars(select(TrainingJob).order_by(TrainingJob.created_at.desc())))
    return _jobs_to_read(db, jobs)


@router.get("/jobs/{job_id}", response_model=TrainingJobRead)
def get_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJobRead:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_read(db, job)


@router.post("/jobs", response_model=TrainingJobRead)
def create_job(payload: TrainingJobCreate, db: Session = Depends(get_db)) -> TrainingJob:
    exp = db.get(Experiment, payload.experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Project not found")

    cfg = _build_config(payload)
    job = TrainingJob(
        experiment_id=payload.experiment_id,
        name=payload.name,
        model_name=cfg.to_train_dict(job_id=0, artifact_dir=".")["model_name"],
        num_workers=payload.num_workers,
        gpus_per_worker=payload.gpus_per_worker,
        epochs=payload.epochs,
        batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        config_json=cfg.to_json(),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    enqueue_training_job(job.id)
    return _job_to_read(db, job)


@router.post("/jobs/{job_id}/scale", response_model=WorkerGroupRead)
def scale_workers(
    job_id: int, payload: ScaleWorkersRequest, db: Session = Depends(get_db)
) -> WorkerGroupRead:
    try:
        wg = scale_job_workers(job_id, payload.target_workers, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return wg


@router.delete("/jobs/{job_id}")
def remove_job(job_id: int, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        delete_training_job(db, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "deleted", "job_id": str(job_id)}


@router.get("/jobs/{job_id}/logs", response_model=JobLogsRead)
def job_logs(job_id: int, db: Session = Depends(get_db)) -> JobLogsRead:
    payload = build_job_logs(db, job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobLogsRead(**payload)


@router.get("/jobs/{job_id}/metrics")
def job_metrics(job_id: int, db: Session = Depends(get_db)) -> dict:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.metrics_json:
        return {"metrics": None}
    return {"metrics": json.loads(job.metrics_json)}
