"""Load training texts (and optional labels) from configured dataset sources."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT
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


def _load_jsonl(path: Path, text_field: str, label_field: str | None) -> tuple[list[str], list[int] | None]:
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


def _load_huggingface(cfg: TrainingDatasetConfig) -> tuple[list[str], list[int] | None]:
    from datasets import load_dataset

    assert cfg.hf_dataset
    repo, inline_config = _parse_hf_dataset_id(cfg.hf_dataset)
    config_name = (cfg.hf_config or "").strip() or inline_config
    if config_name:
        ds = load_dataset(repo, config_name, split=cfg.hf_split)
    else:
        try:
            ds = load_dataset(repo, split=cfg.hf_split)
        except ValueError as exc:
            msg = str(exc)
            if "Config name is missing" in msg or "pick one among the available configs" in msg:
                raise ValueError(
                    f"Hugging Face dataset '{repo}' requires a config name "
                    f"(e.g. wikitext-103-raw-v1). Set dataset.hf_config or use "
                    f"'{repo}:wikitext-103-raw-v1'. Original error: {msg}"
                ) from exc
            raise
    limit = cfg.max_samples or len(ds)
    texts: list[str] = []
    labels: list[int] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        if cfg.hf_text_column not in row:
            raise ValueError(f"HF column '{cfg.hf_text_column}' not in dataset row")
        texts.append(str(row[cfg.hf_text_column]))
        if cfg.hf_label_column:
            labels.append(int(row[cfg.hf_label_column]))
    return texts, labels if cfg.hf_label_column else None


def load_training_samples(
    dataset_cfg: dict[str, Any] | TrainingDatasetConfig,
    *,
    task_type: str,
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
        texts, labels = _load_jsonl(path, cfg.text_field, cfg.label_field)
    elif cfg.source == "huggingface":
        texts, labels = _load_huggingface(cfg)
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
