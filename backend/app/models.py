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


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    experiment: Mapped["Experiment"] = relationship(back_populates="jobs")
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


class PlatformSettings(Base):
    """Singleton (id=1) Ray + CDSW worker defaults for the platform."""

    __tablename__ = "platform_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    ray_local_mode: Mapped[bool] = mapped_column(default=True)
    ray_head_address: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ray_train_num_workers: Mapped[int] = mapped_column(Integer, default=0)
    worker_cpu: Mapped[float] = mapped_column(Float, default=4.0)
    worker_memory_gb: Mapped[float] = mapped_column(Float, default=16.0)
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
