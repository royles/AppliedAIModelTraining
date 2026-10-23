"""Load training texts (and optional labels) from configured dataset sources."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[int, int | None], None]

from app.config import REPO_ROOT
from app.training.corpus_staging import (
    append_corpus_jsonl_line,
    corpus_stage_mode,
    in_memory_row_cap,
    large_file_stage_bytes,
    write_corpus_meta,
)
from app.training.dataset_config import TrainingDatasetConfig
from app.training.text_normalize import normalize_training_text

logger = logging.getLogger(__name__)

DATASETS_ROOT = (REPO_ROOT / "data" / "datasets").resolve()

BUILTIN_DATASET_CATALOG: dict[str, dict[str, str]] = {
    "platform_demo": {
        "title": "Platform demo (Ray / CDSW)",
        "description": "Small built-in Q&A or sentences for smoke tests.",
    },
}


def _builtin_texts(name: str, task_type: str) -> list[str]:
    if name != "platform_demo":
        raise ValueError(f"Unknown builtin dataset: {name}")
    if task_type == "causal_lm":
        return [
            "User: How does Ray scale training?\nAssistant: Ray shards workers across GPUs.",
            "User: What is LoRA?\nAssistant: Low-rank adapters that fine-tune efficiently.",
            "User: Explain CDSW workers.\nAssistant: They launch elastic engines on Cloudera AI.",
        ]
    return [
        "ray scales transformer training across GPUs",
        "cdsw workers attach elastic compute to the cluster",
        "local development uses CPU ray workers",
        "cloudera machine learning hosts the control plane",
    ]


def _resolve_dataset_path(rel_path: str) -> Path:
    clean = rel_path.strip().lstrip("/")
    if ".." in Path(clean).parts:
        raise ValueError("dataset.path must stay under data/datasets")
    full = (DATASETS_ROOT / clean).resolve()
    if not str(full).startswith(str(DATASETS_ROOT)):
        raise ValueError("dataset.path must stay under data/datasets")
    if not full.is_file():
        raise FileNotFoundError(f"Dataset file not found: data/datasets/{clean}")
    return full


def _load_text_file(path: Path, *, task_type: str = "causal_lm") -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Dataset file must be UTF-8 text: {path.name} ({exc})",
        ) from exc
    normalized = normalize_training_text(raw)
    blocks = [b.strip() for b in normalized.split("\n\n") if b.strip()]
    if len(blocks) > 1:
        return blocks
    if blocks and "\n" in blocks[0] and blocks[0].lstrip().startswith("User:"):
        return blocks
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    if task_type == "causal_lm" and lines:
        return _group_user_assistant_lines(lines)
    return blocks or lines


def _group_user_assistant_lines(lines: list[str]) -> list[str]:
    """Pair User:/Assistant: lines into one training example per turn."""
    samples: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("User:") and i + 1 < len(lines) and lines[i + 1].startswith(
            "Assistant:"
        ):
            samples.append(f"{line}\n{lines[i + 1]}")
            i += 2
        else:
            samples.append(line)
            i += 1
    return samples


def _load_jsonl(
    path: Path,
    text_field: str,
    label_field: str | None,
    *,
    progress_cb: ProgressCallback | None = None,
    progress_every: int = 5000,
) -> tuple[list[str], list[int] | None]:
    texts: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if text_field not in row:
                raise ValueError(f"JSONL line {line_no} missing field '{text_field}'")
            texts.append(str(row[text_field]))
            if label_field:
                if label_field not in row:
                    raise ValueError(f"JSONL line {line_no} missing field '{label_field}'")
                labels.append(int(row[label_field]))
            if progress_cb and len(texts) % progress_every == 0:
                progress_cb(len(texts), None)
    if progress_cb and texts:
        progress_cb(len(texts), None)
    return texts, labels if label_field else None


def _parse_hf_dataset_id(raw: str) -> tuple[str, str | None]:
    """
    Allow ``org/name:config`` or ``org/name@config`` in the dataset id field.
    """
    text = raw.strip()
    for sep in (":", "@"):
        if sep in text:
            repo, config = text.split(sep, 1)
            repo = repo.strip()
            config = config.strip() or None
            if repo and config:
                return repo, config
    return text, None


def _open_hf_split(cfg: TrainingDatasetConfig, *, streaming: bool):
    from datasets import load_dataset

    assert cfg.hf_dataset
    repo, inline_config = _parse_hf_dataset_id(cfg.hf_dataset)
    config_name = (cfg.hf_config or "").strip() or inline_config
    kwargs: dict[str, Any] = {"split": cfg.hf_split, "streaming": streaming}
    if config_name:
        return load_dataset(repo, config_name, **kwargs)
    try:
        return load_dataset(repo, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "Config name is missing" in msg or "pick one among the available configs" in msg:
            raise ValueError(
                f"Hugging Face dataset '{repo}' requires a config name "
                f"(e.g. wikitext-103-raw-v1). Set dataset.hf_config or use "
                f"'{repo}:wikitext-103-raw-v1'. Original error: {msg}"
            ) from exc
        raise


def _collect_hf_rows(
    cfg: TrainingDatasetConfig,
    ds,
    *,
    limit: int | None,
    progress_cb: ProgressCallback | None = None,
    progress_every: int = 5000,
) -> tuple[list[str], list[int] | None]:
    texts: list[str] = []
    labels: list[int] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        if cfg.hf_text_column not in row:
            raise ValueError(f"HF column '{cfg.hf_text_column}' not in dataset row")
        texts.append(str(row[cfg.hf_text_column]))
        if cfg.hf_label_column:
            labels.append(int(row[cfg.hf_label_column]))
        n = i + 1
        if progress_cb and n % progress_every == 0:
            progress_cb(n, limit)
    if progress_cb and texts:
        progress_cb(len(texts), limit)
    return texts, labels if cfg.hf_label_column else None


def _load_huggingface(
    cfg: TrainingDatasetConfig,
    *,
    progress_cb: ProgressCallback | None = None,
    progress_every: int = 5000,
) -> tuple[list[str], list[int] | None]:
    limit = cfg.max_samples
    if limit is not None:
        return _load_huggingface_capped(
            cfg, limit, progress_cb=progress_cb, progress_every=progress_every
        )

    ds = _open_hf_split(cfg, streaming=False)
    return _collect_hf_rows(
        cfg,
        ds,
        limit=len(ds),
        progress_cb=progress_cb,
        progress_every=progress_every,
    )


def _load_huggingface_capped(
    cfg: TrainingDatasetConfig,
    limit: int,
    *,
    progress_cb: ProgressCallback | None = None,
    progress_every: int = 5000,
) -> tuple[list[str], list[int] | None]:
    """
    Load at most ``limit`` rows without downloading every HF shard.

    Prefer split slicing (``train[:N]``); fall back to streaming iteration.
    """
    from datasets import load_dataset

    assert cfg.hf_dataset
    repo, inline_config = _parse_hf_dataset_id(cfg.hf_dataset)
    config_name = (cfg.hf_config or "").strip() or inline_config
    base_split = (cfg.hf_split or "train").strip() or "train"
    sliced_split = f"{base_split}[:{limit}]"

    try:
        if config_name:
            ds = load_dataset(repo, config_name, split=sliced_split)
        else:
            ds = load_dataset(repo, split=sliced_split)
        return _collect_hf_rows(
            cfg, ds, limit=None, progress_cb=progress_cb, progress_every=progress_every
        )
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "HF capped load via split slice failed (%s); falling back to streaming",
            exc,
        )

    ds = _open_hf_split(cfg, streaming=True)
    return _collect_hf_rows(
        cfg, ds, limit=limit, progress_cb=progress_cb, progress_every=progress_every
    )


def should_stage_corpus_on_disk(
    dataset_cfg: dict[str, Any] | TrainingDatasetConfig,
    *,
    task_type: str,
) -> bool:
    """Large Hugging Face / file datasets must stream to disk on the driver session."""
    cfg = (
        dataset_cfg
        if isinstance(dataset_cfg, TrainingDatasetConfig)
        else TrainingDatasetConfig.model_validate(dataset_cfg)
    )
    mode = corpus_stage_mode()
    if mode == "memory":
        return False
    if mode == "always":
        return True
    cap = in_memory_row_cap()
    if cfg.source == "huggingface":
        if cfg.max_samples is None:
            return True
        return int(cfg.max_samples) > cap
    if cfg.source in ("jsonl_file", "text_file"):
        try:
            path = _resolve_dataset_path(cfg.path or "")
        except (ValueError, FileNotFoundError):
            return False
        if path.stat().st_size >= large_file_stage_bytes():
            return True
    if cfg.source == "inline" and cfg.texts and len(cfg.texts) > cap:
        return True
    return False


def _prepare_row_for_causal_lm(text: str, *, source: str) -> list[str]:
    from app.training.lm_text_utils import prepare_causal_lm_texts

    return prepare_causal_lm_texts([text], source=source)


def stream_dataset_to_corpus_jsonl(
    dataset_cfg: dict[str, Any] | TrainingDatasetConfig,
    *,
    task_type: str,
    jsonl_path: Path,
    meta_path: Path,
    progress_cb: ProgressCallback | None = None,
    progress_every: int | None = None,
) -> int:
    """
    Download / read dataset rows once, append training examples to ``jsonl_path``.

    Returns total example count written (after causal-LM chunking when applicable).
    """
    cfg = (
        dataset_cfg
        if isinstance(dataset_cfg, TrainingDatasetConfig)
        else TrainingDatasetConfig.model_validate(dataset_cfg)
    )
    every = progress_every if progress_every is not None else _progress_every()
    if jsonl_path.is_file():
        jsonl_path.unlink()
    rows_read = 0
    examples_written = 0
    limit = cfg.max_samples

    def _emit_prepared(texts: list[str], label: int | None = None) -> None:
        nonlocal examples_written
        for t in texts:
            append_corpus_jsonl_line(jsonl_path, t, label=label)
            examples_written += 1

    if cfg.source == "huggingface":
        ds = _open_hf_split(cfg, streaming=True)
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            if cfg.hf_text_column not in row:
                raise ValueError(f"HF column '{cfg.hf_text_column}' not in dataset row")
            raw = str(row[cfg.hf_text_column])
            label = int(row[cfg.hf_label_column]) if cfg.hf_label_column else None
            rows_read += 1
            if task_type == "causal_lm":
                _emit_prepared(_prepare_row_for_causal_lm(raw, source=cfg.source), label)
            else:
                _emit_prepared([raw], label)
            if progress_cb and rows_read % every == 0:
                progress_cb(rows_read, limit)
        if progress_cb and rows_read:
            progress_cb(rows_read, limit)
    elif cfg.source == "jsonl_file":
        path = _resolve_dataset_path(cfg.path or "")
        with path.open(encoding="utf-8") as fp:
            for line_no, raw in enumerate(fp, start=1):
                if limit is not None and rows_read >= limit:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if cfg.text_field not in row:
                    raise ValueError(f"JSONL line {line_no} missing field '{cfg.text_field}'")
                text = str(row[cfg.text_field])
                label = int(row[cfg.label_field]) if cfg.label_field else None
                rows_read += 1
                if task_type == "causal_lm":
                    _emit_prepared(_prepare_row_for_causal_lm(text, source=cfg.source), label)
                else:
                    _emit_prepared([text], label)
                if progress_cb and rows_read % every == 0:
                    progress_cb(rows_read, limit)
    elif cfg.source == "text_file":
        path = _resolve_dataset_path(cfg.path or "")
        raw = path.read_text(encoding="utf-8")
        blocks = _load_text_file(path, task_type=task_type)
        for block in blocks:
            if limit is not None and rows_read >= limit:
                break
            rows_read += 1
            if task_type == "causal_lm":
                _emit_prepared(_prepare_row_for_causal_lm(block, source=cfg.source))
            else:
                _emit_prepared([block])
            if progress_cb and rows_read % every == 0:
                progress_cb(rows_read, limit)
    elif cfg.source == "inline":
        for text in cfg.texts:
            if limit is not None and rows_read >= limit:
                break
            rows_read += 1
            if task_type == "causal_lm":
                _emit_prepared(_prepare_row_for_causal_lm(str(text), source=cfg.source))
            else:
                _emit_prepared([str(text)])
    elif cfg.source == "builtin":
        for text in _builtin_texts(cfg.builtin_name, task_type):
            rows_read += 1
            _emit_prepared([text])
    else:
        raise ValueError(
            f"Disk staging not implemented for dataset source={cfg.source!r}; "
            "use max_samples or a smaller dataset on the session."
        )

    if examples_written == 0:
        raise ValueError("Dataset produced zero training examples")

    write_corpus_meta(meta_path, total_examples=examples_written, source=cfg.source)
    logger.info(
        "Staged corpus on disk rows_read=%s examples=%s path=%s",
        rows_read,
        examples_written,
        jsonl_path,
    )
    return examples_written


def _progress_every() -> int:
    import os

    raw = os.environ.get("PLATFORM_DOWNLOAD_PROGRESS_EVERY", "5000").strip()
    try:
        return max(500, int(raw))
    except ValueError:
        return 5000


def load_training_samples(
    dataset_cfg: dict[str, Any] | TrainingDatasetConfig,
    *,
    task_type: str,
    for_preview: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> tuple[list[str], list[int] | None]:
    cfg = (
        dataset_cfg
        if isinstance(dataset_cfg, TrainingDatasetConfig)
        else TrainingDatasetConfig.model_validate(dataset_cfg)
    )
    texts: list[str]
    labels: list[int] | None = None

    if cfg.source == "builtin":
        texts = _builtin_texts(cfg.builtin_name, task_type)
    elif cfg.source == "inline":
        texts = list(cfg.texts)
        labels = cfg.labels
    elif cfg.source == "text_file":
        path = _resolve_dataset_path(cfg.path or "")
        texts = _load_text_file(path, task_type=task_type)
    elif cfg.source == "jsonl_file":
        path = _resolve_dataset_path(cfg.path or "")
        every = _progress_every()
        texts, labels = _load_jsonl(
            path,
            cfg.text_field,
            cfg.label_field,
            progress_cb=progress_cb,
            progress_every=every,
        )
    elif cfg.source == "huggingface":
        every = _progress_every()
        texts, labels = _load_huggingface(
            cfg, progress_cb=progress_cb, progress_every=every
        )
    elif cfg.source == "iceberg":
        from app.training.iceberg_loader import load_iceberg_samples

        texts, labels = load_iceberg_samples(cfg)
    elif cfg.source == "parquet_s3":
        from app.training.parquet_loader import load_parquet_samples

        texts, labels = load_parquet_samples(cfg)
    else:
        raise ValueError(f"Unsupported dataset source: {cfg.source}")

    if not texts:
        raise ValueError("Dataset produced zero training examples")

    if cfg.max_samples and len(texts) > cfg.max_samples:
        texts = texts[: cfg.max_samples]
        if labels is not None:
            labels = labels[: cfg.max_samples]

    if cfg.repeat > 1:
        texts = texts * cfg.repeat
        if labels is not None:
            labels = labels * cfg.repeat

    if task_type == "sequence_classification" and labels is None:
        labels = [i % 2 for i in range(len(texts))]

    if task_type == "causal_lm":
        from app.training.lm_text_utils import prepare_causal_lm_texts

        before = len(texts)
        prep_kwargs: dict[str, Any] = {"source": cfg.source}
        if for_preview:
            prep_kwargs["min_chunk_chars"] = 1
        texts = prepare_causal_lm_texts(texts, **prep_kwargs)
        if len(texts) != before:
            labels = None
            logger.info(
                "Causal LM text pipeline: %s rows -> %s training examples (chunk/normalize)",
                before,
                len(texts),
            )

    logger.info(
        "Loaded dataset source=%s examples=%s task=%s",
        cfg.source,
        len(texts),
        task_type,
    )
    return texts, labels


def list_dataset_files() -> list[dict[str, str]]:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, str]] = []
    for path in sorted(DATASETS_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".jsonl", ".json"}:
            continue
        rel = path.relative_to(DATASETS_ROOT).as_posix()
        kind = "jsonl_file" if path.suffix.lower() == ".jsonl" else "text_file"
        out.append({"path": rel, "kind": kind, "size_bytes": str(path.stat().st_size)})
    return out
