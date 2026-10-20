import os
from unittest.mock import patch

import pytest

from app.training.dataset_config import TrainingDatasetConfig


def test_iceberg_requires_table():
    with pytest.raises(ValueError, match="iceberg_table"):
        TrainingDatasetConfig(source="iceberg", iceberg_table=None)


def test_iceberg_accepts_env_catalog_uri():
    with patch.dict(os.environ, {"PLATFORM_ICEBERG_CATALOG_URI": "http://catalog:8181"}, clear=False):
        cfg = TrainingDatasetConfig(
            source="iceberg",
            iceberg_table="db.tbl",
            iceberg_text_column="body",
        )
    assert cfg.iceberg_table == "db.tbl"


def test_parquet_s3_requires_uri():
    with pytest.raises(ValueError, match="parquet_uri"):
        TrainingDatasetConfig(source="parquet_s3", parquet_uri=None)


def test_parquet_s3_valid():
    cfg = TrainingDatasetConfig(
        source="parquet_s3",
        parquet_uri="s3://bucket/data/train/",
        parquet_text_column="content",
    )
    assert cfg.parquet_uri.startswith("s3://")
