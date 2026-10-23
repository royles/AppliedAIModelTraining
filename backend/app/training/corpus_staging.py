"""On-disk corpus staging (JSONL) to avoid loading multi-million-row datasets into RAM."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CORPUS_JSONL = "corpus.jsonl"
CORPUS_META = "corpus_meta.json"
PROMPT_FORMAT_PROBE_ROWS = 512


def corpus_stage_mode() -> str:
    """
    ``auto`` (default): stream uncapped / large HF and big files to disk (API-safe).
    ``always``: always use ``corpus.jsonl`` under the job artifact.
    ``memory``: keep rows in RAM when loading (use only when the session has headroom).
    """
    return os.environ.get("PLATFORM_CORPUS_STAGE_MODE", "auto").strip().lower()


def in_memory_row_cap() -> int:
    raw = os.environ.get("PLATFORM_CORPUS_IN_MEMORY_MAX_ROWS", "25000").strip()
    try:
        return max(1000, int(raw))
    except ValueError:
        return 25000


def large_file_stage_bytes() -> int:
    raw = os.environ.get("PLATFORM_CORPUS_STAGE_FILE_BYTES", str(32 * 1024 * 1024)).strip()
    try:
        return max(1024 * 1024, int(raw))
    except ValueError:
        return 32 * 1024 * 1024


@dataclass(frozen=True)
class StagedCorpusRef:
    """Corpus lives on disk; ``texts`` is empty when ``on_disk_only``."""

    texts: list[str]
    labels: list[int] | None
    jsonl_path: Path | None
    example_count: int
    on_disk_only: bool


def write_corpus_meta(path: Path, *, total_examples: int, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"format": "jsonl", "total_examples": total_examples, "source": source},
            indent=2,
        ),
        encoding="utf-8",
    )


def read_corpus_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def append_corpus_jsonl_line(path: Path, text: str, *, label: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {"text": text}
    if label is not None:
        row["label"] = label
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_corpus_jsonl(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid corpus JSONL at line {line_no}: {exc}") from exc
            if not isinstance(row, dict) or "text" not in row:
                raise ValueError(f"Corpus JSONL line {line_no} missing 'text'")
            yield str(row["text"])


def iter_corpus_jsonl_sample(path: Path, max_rows: int) -> list[str]:
    out: list[str] = []
    for text in iter_corpus_jsonl(path):
        out.append(text)
        if len(out) >= max_rows:
            break
    return out


def count_corpus_jsonl_chars(path: Path) -> tuple[int, int]:
    total_chars = 0
    rows = 0
    for text in iter_corpus_jsonl(path):
        total_chars += len(text)
        rows += 1
    return rows, total_chars


def iter_shard_jsonl(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid shard JSONL at line {line_no}: {exc}") from exc
            if not isinstance(row, dict) or "text" not in row:
                raise ValueError(f"Shard JSONL line {line_no} missing 'text'")
            yield str(row["text"])
