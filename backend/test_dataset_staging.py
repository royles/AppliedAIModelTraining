"""Shared data collections and remote vs local staging."""

from pathlib import Path

from app.training.dataset_config import is_remote_dataset
from app.training.dataset_sharding import prepare_dataset_shards_on_driver, shard_cache_dir
from app.training.dataset_staging import (
    auto_data_collection_id,
    create_data_collection,
    ensure_auto_data_collection,
    find_collection_for_dataset,
    register_job_dataset_in_data_collection,
    stage_data_collection,
)


def test_remote_collection_registers_without_corpus_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.training.dataset_staging.COLLECTIONS_ROOT",
        tmp_path / "collections",
    )
    created = create_data_collection(name="Lakehouse")
    cid = created["collection_id"]
    cfg = {
        "source": "parquet_s3",
        "parquet_uri": "s3://bucket/train/",
        "parquet_text_column": "text",
    }
    assert is_remote_dataset(cfg)
    result = stage_data_collection(
        collection_id=cid,
        dataset=cfg,
        task_type="causal_lm",
        ray_train_workers=None,
    )
    assert result["access_mode"] == "remote"
    assert result["remote_uri"] == "s3://bucket/train/"
    assert result["local_path"]
    assert result["corpus_local_path"] is None
    assert not Path(result["cache_local_path"]).joinpath("corpus.json").is_file()

    hit = find_collection_for_dataset(cfg, "causal_lm", preferred_collection_id=cid)
    assert hit is not None
    assert hit["access_mode"] == "remote"


def test_auto_register_syncs_corpus_and_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.training.dataset_staging.COLLECTIONS_ROOT",
        tmp_path / "collections",
    )
    artifact = tmp_path / "job_9"
    artifact.mkdir()
    cfg = {"source": "inline", "texts": ["a", "b", "c", "d"]}
    prepare_dataset_shards_on_driver(
        artifact_dir=str(artifact),
        dataset_cfg=cfg,
        task_type="causal_lm",
        world_size=2,
    )
    ensure_auto_data_collection(project_id=42, project_name="Demo")
    train_config = {
        "dataset": cfg,
        "task_type": "causal_lm",
        "artifact_dir": str(artifact),
        "dataset_shards_prepared": True,
        "ray_train_workers_scheduled": 2,
    }
    reg = register_job_dataset_in_data_collection(
        project_id=42,
        project_name="Demo",
        job_id=9,
        train_config=train_config,
    )
    assert reg is not None
    assert reg["collection_id"] == auto_data_collection_id(42)
    assert reg["reference_only"] is True
    assert reg["shards_saved"] is True
    assert reg["shard_world_size"] == 2
    assert reg["staging_root_ref"] == str(artifact.resolve())
    coll_root = tmp_path / "collections" / auto_data_collection_id(42)
    coll_cache = shard_cache_dir(coll_root, cfg, "causal_lm")
    assert not (coll_cache / "corpus.json").is_file()
    job_cache = shard_cache_dir(artifact, cfg, "causal_lm")
    assert (job_cache / "corpus.json").is_file()
    assert (job_cache / "shard_0.json").is_file()
    assert (job_cache / "shard_1.json").is_file()
    assert train_config.get("data_staging_root_ref") == str(artifact.resolve())
    assert train_config.get("dataset_corpus_ref") == str(job_cache / "corpus.json")
    assert train_config.get("data_collection_id") == auto_data_collection_id(42)
