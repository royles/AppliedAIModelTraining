"""Disk-backed corpus staging for large Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from app.training.corpus_staging import CORPUS_JSONL, CORPUS_META, iter_corpus_jsonl
from app.training.dataset_loader import should_stage_corpus_on_disk, stream_dataset_to_corpus_jsonl
from app.training.dataset_sharding import (
    load_training_corpus_staged,
    shard_cache_dir,
    staged_corpus_ref,
)


def test_should_stage_uncapped_huggingface_by_default():
    cfg = {"source": "huggingface", "hf_dataset": "org/big", "hf_split": "train"}
    assert should_stage_corpus_on_disk(cfg, task_type="causal_lm") is True


def test_should_stage_memory_mode_opt_out(monkeypatch):
    monkeypatch.setenv("PLATFORM_CORPUS_STAGE_MODE", "memory")
    cfg = {"source": "huggingface", "hf_dataset": "org/big", "hf_split": "train"}
    assert should_stage_corpus_on_disk(cfg, task_type="causal_lm") is False


def test_stream_inline_to_jsonl(tmp_path: Path):
    jsonl = tmp_path / "corpus.jsonl"
    meta = tmp_path / "corpus_meta.json"
    cfg = {"source": "inline", "texts": ["hello world", "User: hi\nAssistant: there"]}
    n = stream_dataset_to_corpus_jsonl(
        cfg,
        task_type="causal_lm",
        jsonl_path=jsonl,
        meta_path=meta,
    )
    assert n >= 2
    lines = list(iter_corpus_jsonl(jsonl))
    assert len(lines) == n
    assert meta.is_file()


def test_staged_corpus_reuse_skips_loader(tmp_path: Path):
    artifact = tmp_path / "job_1"
    artifact.mkdir()
    cfg = {"source": "inline", "texts": ["a", "b"]}
    cache = shard_cache_dir(artifact, cfg, "causal_lm")
    jsonl = cache / CORPUS_JSONL
    meta = cache / CORPUS_META
    jsonl.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")
    meta.write_text(
        json.dumps({"format": "jsonl", "total_examples": 2, "source": "inline"}),
        encoding="utf-8",
    )
    with patch("app.training.dataset_sharding.load_training_samples") as load_mock:
        texts, labels = load_training_corpus_staged(
            artifact_dir=str(artifact),
            dataset_cfg=cfg,
            task_type="causal_lm",
        )
    load_mock.assert_not_called()
    assert texts == []
    assert labels is None
    ref = staged_corpus_ref(
        artifact_dir=str(artifact),
        dataset_cfg=cfg,
        task_type="causal_lm",
    )
    assert ref is not None and ref.example_count == 2
