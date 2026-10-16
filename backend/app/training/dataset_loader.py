"""Load training texts (and optional labels) from configured dataset sources."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT
from app.training.dataset_config import TrainingDatasetConfig

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


def _load_text_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


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


def _load_huggingface(cfg: TrainingDatasetConfig) -> tuple[list[str], list[int] | None]:
    from datasets import load_dataset

    assert cfg.hf_dataset
    ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
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
        texts = _load_text_file(path)
    elif cfg.source == "jsonl_file":
        path = _resolve_dataset_path(cfg.path or "")
        texts, labels = _load_jsonl(path, cfg.text_field, cfg.label_field)
    elif cfg.source == "huggingface":
        texts, labels = _load_huggingface(cfg)
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
