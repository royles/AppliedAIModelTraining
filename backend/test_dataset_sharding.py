"""Dataset sharding for distributed Ray Train."""

import json
from pathlib import Path
from unittest.mock import patch

from app.training.dataset_sharding import (
    load_dataset_shard_for_worker,
    load_training_corpus_staged,
    prepare_dataset_shards_on_driver,
    prepared_dataset_shards_meta,
    shard_cache_dir,
    split_interleaved,
)


def test_split_interleaved_even():
    texts = [f"t{i}" for i in range(8)]
    labels = list(range(8))
    shards = split_interleaved(texts, labels, 4)
    assert [s[0] for s in shards] == [["t0", "t4"], ["t1", "t5"], ["t2", "t6"], ["t3", "t7"]]
    assert [s[1] for s in shards] == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_driver_shards_and_worker_load(tmp_path: Path):
    artifact = tmp_path / "job_1"
    artifact.mkdir()
    cfg = {"source": "inline", "texts": ["a", "b", "c", "d", "e"]}
    meta = prepare_dataset_shards_on_driver(
        artifact_dir=str(artifact),
        dataset_cfg=cfg,
        task_type="causal_lm",
        world_size=2,
    )
    assert meta["total_examples"] == 5
    assert meta["examples_per_shard"] == [3, 2]

    t0, _ = load_dataset_shard_for_worker(
        artifact_dir=str(artifact),
        dataset_cfg=cfg,
        task_type="causal_lm",
        rank=0,
        world_size=2,
    )
    t1, _ = load_dataset_shard_for_worker(
        artifact_dir=str(artifact),
        dataset_cfg=cfg,
        task_type="causal_lm",
        rank=1,
        world_size=2,
    )
    assert t0 == ["a", "c", "e"]
    assert t1 == ["b", "d"]

    cache = shard_cache_dir(artifact, cfg, "causal_lm")
    ready = json.loads((cache / "ready.json").read_text(encoding="utf-8"))
    assert ready["world_size"] == 2


def test_prepared_meta_requires_all_shard_files(tmp_path: Path):
    artifact = tmp_path / "job_2"
    artifact.mkdir()
    cfg = {"source": "inline", "texts": ["x", "y"]}
    cache = shard_cache_dir(artifact, cfg, "causal_lm")
    (cache / "ready.json").write_text(
        json.dumps({"world_size": 2, "total_examples": 2}),
        encoding="utf-8",
    )
    assert prepared_dataset_shards_meta(
        artifact_dir=str(artifact),
        dataset_cfg=cfg,
        task_type="causal_lm",
        world_size=2,
    ) is None

    (cache / "shard_0.json").write_text(json.dumps({"texts": ["x"]}), encoding="utf-8")
    (cache / "shard_1.json").write_text(json.dumps({"texts": ["y"]}), encoding="utf-8")
    meta = prepared_dataset_shards_meta(
        artifact_dir=str(artifact),
        dataset_cfg=cfg,
        task_type="causal_lm",
        world_size=2,
    )
    assert meta is not None
    assert meta["total_examples"] == 2


def test_staged_corpus_skips_loader(tmp_path: Path):
    artifact = tmp_path / "job_3"
    artifact.mkdir()
    cfg = {"source": "inline", "texts": ["hello", "world"]}
    cache = shard_cache_dir(artifact, cfg, "causal_lm")
    (cache / "corpus.json").write_text(
        json.dumps({"texts": ["hello", "world"]}),
        encoding="utf-8",
    )
    with patch("app.training.dataset_sharding.load_training_samples") as load_mock:
        texts, labels = load_training_corpus_staged(
            artifact_dir=str(artifact),
            dataset_cfg=cfg,
            task_type="causal_lm",
        )
    load_mock.assert_not_called()
    assert texts == ["hello", "world"]
    assert labels is None


def test_prepare_shards_reuses_disk_without_reload(tmp_path: Path):
    artifact = tmp_path / "job_4"
    artifact.mkdir()
    cfg = {"source": "inline", "texts": ["a", "b", "c"]}
    with patch("app.training.dataset_sharding.load_training_samples") as load_mock:
        load_mock.return_value = (["a", "b", "c"], None)
        prepare_dataset_shards_on_driver(
            artifact_dir=str(artifact),
            dataset_cfg=cfg,
            task_type="causal_lm",
            world_size=2,
        )
        assert load_mock.call_count == 1
        load_mock.reset_mock()
        meta = prepare_dataset_shards_on_driver(
            artifact_dir=str(artifact),
            dataset_cfg=cfg,
            task_type="causal_lm",
            world_size=2,
        )
        load_mock.assert_not_called()
    assert meta["total_examples"] == 3
