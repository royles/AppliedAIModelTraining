from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list["TrainingJob"]] = relationship(back_populates="experiment")
    tasks: Mapped[list["TaskDefinition"]] = relationship(back_populates="project")
    ray_settings: Mapped["ProjectRaySettings | None"] = relationship(
        back_populates="project", uselist=False
    )


class TaskDefinition(Base):
    """Long-lived task identity within a project (configuration is versioned)."""

    __tablename__ = "task_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped["Experiment"] = relationship(back_populates="tasks")
    versions: Mapped[list["TaskVersion"]] = relationship(back_populates="task")


class TaskVersion(Base):
    """Immutable snapshot of a task configuration (append-only spec_json)."""

    __tablename__ = "task_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task_definitions.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    spec_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    task: Mapped["TaskDefinition"] = relationship(back_populates="versions")
    jobs: Mapped[list["TrainingJob"]] = relationship(back_populates="task_version")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    task_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("task_versions.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    model_name: Mapped[str] = mapped_column(String(128), default="distilbert-base-uncased")
    num_workers: Mapped[int] = mapped_column(Integer, default=1)
    gpus_per_worker: Mapped[int] = mapped_column(Integer, default=0)
    epochs: Mapped[int] = mapped_column(Integer, default=1)
    batch_size: Mapped[int] = mapped_column(Integer, default=8)
    learning_rate: Mapped[float] = mapped_column(Float, default=5e-5)
    max_steps: Mapped[int] = mapped_column(Integer, default=50)
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ray_address: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    experiment: Mapped["Experiment"] = relationship(back_populates="jobs")
    task_version: Mapped["TaskVersion | None"] = relationship(back_populates="jobs")
    worker_group: Mapped["WorkerGroup | None"] = relationship(
        back_populates="job", uselist=False
    )


class WorkerGroup(Base):
    """Tracks Cloudera AI / CDSW worker engines attached to a Ray cluster."""

    __tablename__ = "worker_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="provisioning")
    target_workers: Mapped[int] = mapped_column(Integer, default=1)
    worker_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    head_address: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped["TrainingJob"] = relationship(back_populates="worker_group")


class ProjectRaySettings(Base):
    """Ray cluster + CDSW worker targets for one project (tasks/jobs inherit this)."""

    __tablename__ = "project_ray_settings"

    project_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), primary_key=True
    )
    ray_local_mode: Mapped[bool] = mapped_column(default=True)
    ray_head_address: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ray_train_num_workers: Mapped[int] = mapped_column(Integer, default=0)
    non_distributed_low_memory_mode: Mapped[bool] = mapped_column(default=False)
    worker_cpu: Mapped[float] = mapped_column(Float, default=4.0)
    worker_memory_gb: Mapped[float] = mapped_column(Float, default=8.0)
    gpus_per_worker: Mapped[int] = mapped_column(Integer, default=0)
    default_job_num_workers: Mapped[int] = mapped_column(Integer, default=1)
    accelerator_label_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_group_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    project: Mapped["Experiment"] = relationship(back_populates="ray_settings")


class PlatformSettings(Base):
    """Singleton (id=1) Ray + CDSW worker defaults for new projects."""

    __tablename__ = "platform_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    ray_local_mode: Mapped[bool] = mapped_column(default=True)
    ray_head_address: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ray_train_num_workers: Mapped[int] = mapped_column(Integer, default=0)
    non_distributed_low_memory_mode: Mapped[bool] = mapped_column(default=False)
    worker_cpu: Mapped[float] = mapped_column(Float, default=4.0)
    worker_memory_gb: Mapped[float] = mapped_column(Float, default=8.0)
    gpus_per_worker: Mapped[int] = mapped_column(Integer, default=0)
    default_job_num_workers: Mapped[int] = mapped_column(Integer, default=1)
    accelerator_label_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_group_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class TrainedModel(Base):
    """Registered artifact exposed via the OpenAI-compatible /v1 API."""

    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), nullable=False)
    openai_model_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    base_model: Mapped[str] = mapped_column(String(256), nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    architecture_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at_unix: Mapped[int] = mapped_column(Integer, nullable=False)
    lifecycle_state: Mapped[str] = mapped_column(String(32), default="trained")
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_card_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    benchmark_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("trained_models.id"), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
