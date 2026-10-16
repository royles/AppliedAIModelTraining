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
from app.services.job_log import build_job_logs
from app.services.job_runner import enqueue_training_job, scale_job_workers
from app.services.platform_settings import get_platform_settings, update_platform_settings
from app.training.architecture import TransformerArchitectureConfig
from app.training.transformer_config import (
    TransformerTrainingConfig,
    list_presets,
)

router = APIRouter()


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
def api_root() -> dict[str, str]:
    return {
        "service": "applied-ai-model-training",
        "health": "/api/health",
        "openapi": "/docs",
        "note": "OpenAPI /docs is on the API port (PLATFORM_API_PORT), not the Application UX port.",
    }


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


@router.get("/experiments", response_model=list[ExperimentRead])
def list_experiments(db: Session = Depends(get_db)) -> list[Experiment]:
    return list(db.scalars(select(Experiment).order_by(Experiment.created_at.desc())))


@router.post("/experiments", response_model=ExperimentRead)
def create_experiment(payload: ExperimentCreate, db: Session = Depends(get_db)) -> Experiment:
    exp = Experiment(name=payload.name, description=payload.description)
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/jobs", response_model=list[TrainingJobRead])
def list_jobs(db: Session = Depends(get_db)) -> list[TrainingJob]:
    return list(db.scalars(select(TrainingJob).order_by(TrainingJob.created_at.desc())))


@router.get("/jobs/{job_id}", response_model=TrainingJobRead)
def get_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs", response_model=TrainingJobRead)
def create_job(payload: TrainingJobCreate, db: Session = Depends(get_db)) -> TrainingJob:
    exp = db.get(Experiment, payload.experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

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
    return job


@router.post("/jobs/{job_id}/scale", response_model=WorkerGroupRead)
def scale_workers(
    job_id: int, payload: ScaleWorkersRequest, db: Session = Depends(get_db)
) -> WorkerGroupRead:
    try:
        wg = scale_job_workers(job_id, payload.target_workers, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return wg


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
