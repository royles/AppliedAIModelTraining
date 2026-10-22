"""Shared data collections and remote vs local staging."""

from pathlib import Path

from app.training.dataset_config import is_remote_dataset
from app.training.dataset_staging import (
    create_data_collection,
    find_collection_for_dataset,
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
