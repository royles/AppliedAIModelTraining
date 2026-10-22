"""Training data source configuration (stored on each job's transformer config)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DatasetSource = Literal[
    "builtin",
    "inline",
    "text_file",
    "jsonl_file",
    "huggingface",
    "iceberg",
    "parquet_s3",
]


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
    hf_config: str | None = Field(
        default=None,
        description="Dataset config/subset (e.g. wikitext-103-raw-v1 for Salesforce/wikitext).",
    )
    hf_split: str = Field(default="train")
    hf_text_column: str = Field(default="text")
    hf_label_column: str | None = None
    iceberg_catalog_name: str = Field(
        default="training",
        description="PyIceberg catalog name (logical; properties define REST/Hive endpoint).",
    )
    iceberg_catalog_type: str = Field(
        default="rest",
        description="Catalog backend type for PyIceberg (rest, hive, glue, …).",
    )
    iceberg_catalog_uri: str | None = Field(
        default=None,
        description="REST or Hive catalog URI; falls back to PLATFORM_ICEBERG_CATALOG_URI.",
    )
    iceberg_warehouse: str | None = Field(
        default=None,
        description="Optional warehouse path for Hive-style catalogs.",
    )
    iceberg_table: str | None = Field(
        default=None,
        description="Iceberg table identifier namespace.table.",
    )
    iceberg_text_column: str = Field(
        default="text",
        description="String column used as each training example.",
    )
    iceberg_label_column: str | None = Field(
        default=None,
        description="Optional int label column for sequence_classification.",
    )
    parquet_uri: str | None = Field(
        default=None,
        description="s3://bucket/prefix/… Parquet file or directory; or path under data/datasets/.",
    )
    parquet_text_column: str = Field(
        default="text",
        description="String column in Parquet for each training example.",
    )
    parquet_label_column: str | None = Field(
        default=None,
        description="Optional int label column in Parquet.",
    )
    s3_region: str | None = Field(
        default=None,
        description="AWS region for S3 Parquet reads (else AWS_REGION / session default).",
    )
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
        if self.source == "iceberg":
            if not self.iceberg_table:
                raise ValueError("dataset.iceberg_table is required when source=iceberg")
            if not (self.iceberg_catalog_uri or "").strip():
                import os

                if not (
                    os.environ.get("PLATFORM_ICEBERG_CATALOG_URI")
                    or os.environ.get("ICEBERG_REST_URI")
                ):
                    raise ValueError(
                        "dataset.iceberg_catalog_uri or PLATFORM_ICEBERG_CATALOG_URI is required "
                        "when source=iceberg"
                    )
        if self.source == "parquet_s3" and not (self.parquet_uri or "").strip():
            raise ValueError("dataset.parquet_uri is required when source=parquet_s3")
        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError("dataset.labels must match dataset.texts length")
        return self


REMOTE_DATASET_SOURCES = frozenset({"iceberg", "parquet_s3"})


def is_remote_dataset(dataset_cfg: dict[str, Any] | TrainingDatasetConfig) -> bool:
    if isinstance(dataset_cfg, TrainingDatasetConfig):
        return dataset_cfg.source in REMOTE_DATASET_SOURCES
    return str(dataset_cfg.get("source") or "") in REMOTE_DATASET_SOURCES


def remote_data_uri(dataset_cfg: dict[str, Any] | TrainingDatasetConfig) -> str | None:
    cfg = (
        dataset_cfg.model_dump()
        if isinstance(dataset_cfg, TrainingDatasetConfig)
        else dataset_cfg
    )
    source = str(cfg.get("source") or "")
    if source == "iceberg":
        table = (cfg.get("iceberg_table") or "").strip()
        catalog = (cfg.get("iceberg_catalog_uri") or "").strip()
        if table and catalog:
            return f"{catalog} · {table}"
        return table or catalog or None
    if source == "parquet_s3":
        return (cfg.get("parquet_uri") or "").strip() or None
    return None
