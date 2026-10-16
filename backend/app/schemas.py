from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.training.architecture import TransformerArchitectureConfig
from app.training.transformer_config import TransformerTrainingConfig


class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None


class ExperimentRead(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingJobCreate(BaseModel):
    experiment_id: int
    name: str
    preset: str | None = Field(
        default="scratch_gpt2_tiny",
        description="Named preset; merged with transformer overrides.",
    )
    transformer: TransformerTrainingConfig | None = Field(
        default=None,
        description="Full or partial transformer config (architecture, training, init_mode).",
    )
    openai_model_id: str | None = None
    num_workers: int = Field(default=1, ge=1, le=32)
    gpus_per_worker: int = Field(default=0, ge=0, le=8)
    # Legacy flat fields (applied as overrides when transformer is omitted)
    model_name: str | None = None
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int | None = Field(default=None, ge=1, le=256)
    learning_rate: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, ge=1, le=100_000)


class TrainingJobRead(BaseModel):
    id: int
    experiment_id: int
    name: str
    status: str
    model_name: str
    num_workers: int
    gpus_per_worker: int
    epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    config_json: str | None
    ray_address: str | None
    metrics_json: str | None
    error_message: str | None
    openai_model_id: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LogLineRead(BaseModel):
    ts: str
    message: str


class JobLogsRead(BaseModel):
    job_id: int
    status: str
    ray_address: str | None
    framework: list[LogLineRead]
    training: list[LogLineRead]
    ray_status: dict[str, Any]
    metrics: dict[str, Any] | None


class ScaleWorkersRequest(BaseModel):
    target_workers: int = Field(ge=1, le=32)


class WorkerGroupRead(BaseModel):
    id: int
    job_id: int
    status: str
    target_workers: int
    worker_ids_json: str | None
    head_address: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    ray_runtime: str
    cdsw_available: bool
    details: dict[str, Any]


class TransformerPresetRead(BaseModel):
    name: str
    title: str
    description: str
    config: TransformerTrainingConfig


class ArchitectureSummary(BaseModel):
    architecture: TransformerArchitectureConfig
    estimated_parameters: int


class RaySettings(BaseModel):
    ray_local_mode: bool = True
    ray_head_address: str | None = Field(
        default=None,
        description="Override Ray GCS head host:port (e.g. session IP:6379). Empty uses auto-detect.",
    )
    ray_train_num_workers: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Cap Ray Train workers per job; 0 = automatic (low-memory aware).",
    )


class CdswWorkerSettings(BaseModel):
    worker_cpu: float = Field(default=4.0, gt=0, le=128)
    worker_memory_gb: float = Field(default=16.0, ge=1, le=512)
    gpus_per_worker: int = Field(default=0, ge=0, le=8)
    default_job_num_workers: int = Field(default=1, ge=1, le=32)
    accelerator_label_id: str | None = None
    resource_profile_id: str | None = None
    resource_group_id: str | None = None


class PlatformRuntimeInfo(BaseModel):
    ray_runtime: str
    cdsw_available: bool
    detected_ray_head: str | None = None


class PlatformSettingsRead(BaseModel):
    ray: RaySettings
    cdsw_workers: CdswWorkerSettings
    runtime: PlatformRuntimeInfo


class PlatformSettingsUpdate(BaseModel):
    ray: RaySettings | None = None
    cdsw_workers: CdswWorkerSettings | None = None
