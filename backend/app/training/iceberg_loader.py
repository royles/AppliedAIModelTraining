"""Load training rows from an Apache Iceberg table (REST / Hive catalog)."""

from __future__ import annotations

import logging
import os
from typing import Any

from app.training.columnar_samples import texts_and_labels_from_arrow
from app.training.dataset_config import TrainingDatasetConfig

logger = logging.getLogger(__name__)


def _catalog_properties(cfg: TrainingDatasetConfig) -> dict[str, str]:
    uri = (cfg.iceberg_catalog_uri or "").strip() or os.environ.get(
        "PLATFORM_ICEBERG_CATALOG_URI", ""
    ).strip() or os.environ.get("ICEBERG_REST_URI", "").strip()
    if not uri:
        raise ValueError(
            "Iceberg REST catalog URI is required (dataset.iceberg_catalog_uri or "
            "PLATFORM_ICEBERG_CATALOG_URI / ICEBERG_REST_URI in the session)."
        )
    props: dict[str, str] = {
        "uri": uri,
        "type": (cfg.iceberg_catalog_type or "rest").strip() or "rest",
    }
    warehouse = (cfg.iceberg_warehouse or "").strip() or os.environ.get(
        "PLATFORM_ICEBERG_WAREHOUSE", ""
    ).strip()
    if warehouse:
        props["warehouse"] = warehouse
    return props


def _table_identifier(cfg: TrainingDatasetConfig) -> str:
    ident = (cfg.iceberg_table or "").strip()
    if not ident or "." not in ident:
        raise ValueError(
            "dataset.iceberg_table must be namespace.table (e.g. ml.training_corpus)"
        )
    return ident


def load_iceberg_samples(cfg: TrainingDatasetConfig) -> tuple[list[str], list[int] | None]:
    try:
        from pyiceberg.catalog import load_catalog
    except ImportError as exc:
        raise ImportError(
            "pyiceberg is not installed. Add pyiceberg to backend/requirements.txt "
            "or pip install pyiceberg in the project .venv."
        ) from exc

    catalog_name = (cfg.iceberg_catalog_name or "training").strip() or "training"
    props = _catalog_properties(cfg)
    table_id = _table_identifier(cfg)
    text_col = (cfg.iceberg_text_column or "text").strip() or "text"
    label_col = (cfg.iceberg_label_column or "").strip() or None

    logger.info(
        "Loading Iceberg table %s from catalog %s (%s)",
        table_id,
        catalog_name,
        props.get("uri", props.get("type")),
    )
    catalog = load_catalog(catalog_name, **props)
    table = catalog.load_table(table_id)
    limit = int(cfg.max_samples) if cfg.max_samples else None
    scan = table.scan(limit=limit) if limit else table.scan()
    arrow = scan.to_arrow()

    return texts_and_labels_from_arrow(
        arrow,
        text_column=text_col,
        label_column=label_col,
        source_name=f"Iceberg {table_id}",
    )
