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
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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
