"""Training data source configuration (stored on each job's transformer config)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DatasetSource = Literal["builtin", "inline", "text_file", "jsonl_file", "huggingface"]


class TrainingDatasetConfig(BaseModel):
    source: DatasetSource = Field(
        default="builtin",
        description="builtin | inline texts | file under data/datasets | Hugging Face hub",
    )
    builtin_name: str = Field(
        default="platform_demo",
        description="When source=builtin, which packaged demo corpus to use.",
    )
    texts: list[str] = Field(
        default_factory=list,
        description="When source=inline, raw training strings.",
    )
    labels: list[int] | None = Field(
        default=None,
        description="Optional per-text labels for sequence_classification.",
    )
    path: str | None = Field(
        default=None,
        description="Relative path under data/datasets/ for text_file or jsonl_file.",
    )
    text_field: str = Field(default="text", description="JSONL column for text content.")
    label_field: str | None = Field(
        default=None,
        description="JSONL column for classification labels.",
    )
    hf_dataset: str | None = Field(default=None, description="Hugging Face datasets id.")
    hf_split: str = Field(default="train")
    hf_text_column: str = Field(default="text")
    hf_label_column: str | None = None
    max_samples: int | None = Field(default=None, ge=1, le=500_000)
    repeat: int = Field(
        default=1,
        ge=1,
        le=10_000,
        description="Repeat the loaded corpus this many times (small sets).",
    )

    @model_validator(mode="after")
    def validate_source_fields(self) -> TrainingDatasetConfig:
        if self.source == "inline" and not self.texts:
            raise ValueError("dataset.texts is required when source=inline")
        if self.source in ("text_file", "jsonl_file") and not self.path:
            raise ValueError("dataset.path is required for file-based sources")
        if self.source == "huggingface" and not self.hf_dataset:
            raise ValueError("dataset.hf_dataset is required when source=huggingface")
        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError("dataset.labels must match dataset.texts length")
        return self
