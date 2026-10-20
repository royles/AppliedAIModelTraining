"""Extract training texts from PyArrow tables (Parquet / Iceberg scans)."""

from __future__ import annotations

import pyarrow as pa


def texts_and_labels_from_arrow(
    arrow: pa.Table,
    *,
    text_column: str,
    label_column: str | None,
    source_name: str,
) -> tuple[list[str], list[int] | None]:
    if text_column not in arrow.column_names:
        raise ValueError(
            f"Column '{text_column}' not found in {source_name} "
            f"(columns: {', '.join(arrow.column_names)})"
        )
    texts = [str(v) if v is not None else "" for v in arrow[text_column].to_pylist()]
    labels: list[int] | None = None
    if label_column:
        if label_column not in arrow.column_names:
            raise ValueError(
                f"Label column '{label_column}' not found in {source_name}"
            )
        labels = [int(v) for v in arrow[label_column].to_pylist()]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError(f"{source_name} produced zero non-empty text rows")
    return texts, labels
