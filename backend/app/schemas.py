from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.training.architecture import TransformerArchitectureConfig
from app.training.transformer_config import TransformerTrainingConfig


class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None


class ExperimentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None


class ExperimentRead(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TaskDefinitionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class TaskDefinitionRead(BaseModel):
    id: int
    project_id: int
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TaskVersionCreate(BaseModel):
    spec: dict[str, Any] = Field(description="Full model task blueprint JSON")


class TaskVersionRead(BaseModel):
    id: int
    task_id: int
    version: int
    content_hash: str
    spec: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingJobCreate(BaseModel):
    experiment_id: int
    name: str
    task_version_id: int | None = Field(
        default=None,
        description="Run an existing immutable task version.",
    )
    save_as_task: bool = Field(
        default=True,
        description="When task_blueprint is sent, persist a new task version before running.",
    )
    task_name: str | None = Field(
        default=None,
        description="Name for a new task definition when save_as_task is true.",
    )
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
    task_blueprint: dict | None = Field(
        default=None,
        description="Full model-task wizard JSON snapshot saved with the job artifact.",
    )


class JobRestartRequest(BaseModel):
    resume: bool = Field(
        default=True,
        description="When true, continue from progress.json / last epoch checkpoint if present.",
    )


class TrainingJobUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    experiment_id: int | None = None
    openai_model_id: str | None = Field(default=None, max_length=64)


class TrainingJobRead(BaseModel):
    id: int
    experiment_id: int
    project_id: int | None = None
    task_version_id: int | None = None
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

    @model_validator(mode="after")
    def fill_project_id(self) -> "TrainingJobRead":
        if self.project_id is None:
            self.project_id = self.experiment_id
        return self


class UiBuildFeaturesRead(BaseModel):
    model_registry: bool = False
    registry_chat: bool = False


class UiBuildInfoRead(BaseModel):
    status: str
    platform_version: str | None = None
    js: str | None = None
    css: str | None = None
    stamp: str | None = None
    features: UiBuildFeaturesRead | None = None


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
    non_distributed_low_memory_mode: bool = Field(
        default=False,
        description=(
            "Session-only Train: 1 rank, no CDSW worker launch, small batch. "
            "Use when the session is small but you are not using distributed engines."
        ),
    )


class CdswWorkerSettings(BaseModel):
    worker_cpu: float = Field(default=4.0, gt=0, le=128)
    worker_memory_gb: float = Field(default=8.0, ge=8, le=512)
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


class RegistryLifecycleInfo(BaseModel):
    state: str
    guidance: str
    transitions_to: list[str]


class RegistryModelRead(BaseModel):
    id: int
    job_id: int
    job_name: str | None = None
    experiment_id: int | None = Field(
        default=None,
        description="Training project id (experiments table).",
    )
    project_name: str | None = None
    openai_model_id: str
    display_name: str | None
    description: str | None
    lifecycle_state: str
    task_type: str
    base_model: str
    artifact_path: str
    created_at_unix: int
    updated_at: datetime
    parent_model_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] | None = None
    model_card_markdown: str | None = None


class RegistryModelUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    benchmark: dict[str, Any] | None = None
    model_card_markdown: str | None = None


class RegistryTransitionRequest(BaseModel):
    lifecycle_state: str
