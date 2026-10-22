"""Dataset sharding for distributed Ray Train."""

import json
from pathlib import Path

from app.training.dataset_sharding import (
    load_dataset_shard_for_worker,
    prepare_dataset_shards_on_driver,
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
