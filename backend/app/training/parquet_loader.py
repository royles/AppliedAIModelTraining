"""Load training rows from Parquet on S3 (or local file paths for dev)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from app.config import REPO_ROOT
from app.training.columnar_samples import texts_and_labels_from_arrow
from app.training.dataset_config import TrainingDatasetConfig

logger = logging.getLogger(__name__)

DATASETS_ROOT = (REPO_ROOT / "data" / "datasets").resolve()


def _normalize_storage_uri(uri: str) -> str:
    raw = uri.strip()
    if raw.startswith("s3a://"):
        return "s3://" + raw[6:]
    if raw.startswith("s3n://"):
        return "s3://" + raw[6:]
    return raw


def _s3_filesystem(region: str | None):
    import pyarrow.fs as pafs

    reg = (region or "").strip() or os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION"
    )
    kwargs: dict[str, str] = {}
    if reg:
        kwargs["region"] = reg
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT")
    if endpoint:
        kwargs["endpoint_override"] = endpoint
    return pafs.S3FileSystem(**kwargs)


def _resolve_parquet_dataset(cfg: TrainingDatasetConfig):
    import pyarrow.dataset as ds

    uri = _normalize_storage_uri(cfg.parquet_uri or "")
    if not uri:
        raise ValueError("dataset.parquet_uri is required when source=parquet_s3")

    text_col = (cfg.parquet_text_column or "text").strip() or "text"
    label_col = (cfg.parquet_label_column or "").strip() or None
    columns = [text_col]
    if label_col:
        columns.append(label_col)

    limit = int(cfg.max_samples) if cfg.max_samples else None

    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket:
            raise ValueError(f"Invalid S3 URI: {uri}")
        fs = _s3_filesystem(cfg.s3_region)
        path = f"{bucket}/{key}" if key else bucket
        logger.info("Reading Parquet from s3://%s/%s", bucket, key or "")
        dataset = ds.dataset(path, format="parquet", filesystem=fs)
    elif uri.startswith("file://"):
        local = Path(urlparse(uri).path)
        if not local.exists():
            raise FileNotFoundError(f"Parquet path not found: {local}")
        logger.info("Reading Parquet from %s", local)
        dataset = ds.dataset(str(local), format="parquet")
    else:
        rel = uri.lstrip("/")
        if ".." in Path(rel).parts:
            raise ValueError("parquet_uri must be s3://, file://, or under data/datasets/")
        local = (DATASETS_ROOT / rel).resolve()
        if not str(local).startswith(str(DATASETS_ROOT)):
            raise ValueError("parquet_uri must stay under data/datasets when relative")
        if not local.exists():
            raise FileNotFoundError(f"Parquet not found: data/datasets/{rel}")
        logger.info("Reading Parquet from data/datasets/%s", rel)
        dataset = ds.dataset(str(local), format="parquet")

    scanner = dataset.scanner(columns=columns, limit=limit)
    arrow = scanner.to_table()
    return arrow, text_col, label_col, uri


def load_parquet_samples(cfg: TrainingDatasetConfig) -> tuple[list[str], list[int] | None]:
    try:
        import pyarrow.dataset  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Parquet datasets (install backend/requirements.txt)."
        ) from exc

    arrow, text_col, label_col, uri = _resolve_parquet_dataset(cfg)
    return texts_and_labels_from_arrow(
        arrow,
        text_column=text_col,
        label_column=label_col,
        source_name=f"Parquet {uri}",
    )
