"""Shared data collections — staged corpora reused across many training runs."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT
from app.training.dataset_config import (
    TrainingDatasetConfig,
    is_remote_dataset,
    remote_data_uri,
)
from app.training.dataset_sharding import (
    _CORPUS_FILE,
    _READY_FILE,
    _cache_key,
    load_training_corpus_staged,
    prepare_dataset_shards_on_driver,
    prepared_dataset_shards_meta,
    shard_cache_dir,
)

logger = logging.getLogger(__name__)

STAGING_ROOT = (REPO_ROOT / "data" / "staging").resolve()
COLLECTIONS_ROOT = (STAGING_ROOT / "collections").resolve()
_COLLECTION_META = "collection.json"

from app.training.dataset_config import REMOTE_DATASET_SOURCES

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "collection"


def collection_dir(collection_id: str) -> Path:
    root = (COLLECTIONS_ROOT / collection_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def local_path_fields(
    collection_root: str | Path,
    dataset_cfg: dict[str, Any],
    task_type: str,
) -> dict[str, str]:
    """Absolute paths exposed to the UI and job logs."""
    root = Path(collection_root).resolve()
    cache = shard_cache_dir(root, dataset_cfg, task_type)
    corpus = cache / _CORPUS_FILE
    return {
        "local_path": str(root),
        "cache_local_path": str(cache),
        "corpus_local_path": str(corpus),
    }


def _read_collection_meta(path: Path) -> dict[str, Any] | None:
    meta_file = path / _COLLECTION_META
    if not meta_file.is_file():
        return None
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _write_collection_meta(collection_id: str, payload: dict[str, Any]) -> None:
    root = collection_dir(collection_id)
    payload = {**payload, "collection_id": collection_id, "local_path": str(root)}
    (root / _COLLECTION_META).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_data_collection(*, name: str, description: str | None = None) -> dict[str, Any]:
    base = _slugify(name)
    collection_id = base
    n = 2
    while collection_dir(collection_id).joinpath(_COLLECTION_META).is_file():
        collection_id = f"{base}-{n}"
        n += 1
    root = collection_dir(collection_id)
    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "name": name.strip(),
        "description": (description or "").strip() or None,
        "created_at": now,
        "updated_at": now,
    }
    _write_collection_meta(collection_id, meta)
    return {
        "collection_id": collection_id,
        "name": meta["name"],
        "description": meta["description"],
        "local_path": str(root),
        "created_at": now,
        "updated_at": now,
    }


def list_data_collections() -> list[dict[str, Any]]:
    COLLECTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for child in sorted(COLLECTIONS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        meta = _read_collection_meta(child)
        if meta is None:
            continue
        out.append(
            {
                "collection_id": child.name,
                "name": meta.get("name") or child.name,
                "description": meta.get("description"),
                "local_path": str(child.resolve()),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "last_staged": meta.get("last_staged"),
            }
        )
    out.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
    return out


def describe_data_collection(collection_id: str) -> dict[str, Any]:
    root = collection_dir(collection_id)
    meta = _read_collection_meta(root)
    if meta is None:
        raise ValueError(f"Data collection not found: {collection_id}")
    staged = meta.get("last_staged")
    paths: dict[str, str] | None = None
    if isinstance(staged, dict) and staged.get("cache_key"):
        if staged.get("access_mode") == "remote":
            paths = {"local_path": str(root)}
            staged = {**staged, "on_disk": False, "shards_ready": False, **paths}
        else:
            cache_dir = root / ".dataset_shards" / str(staged["cache_key"])
            corpus_path = cache_dir / _CORPUS_FILE
            paths = {
                "local_path": str(root),
                "cache_local_path": str(cache_dir),
                "corpus_local_path": str(corpus_path),
            }
            on_disk = corpus_path.is_file()
            shard_ws = staged.get("shard_world_size")
            shards_ready = False
            if on_disk and isinstance(shard_ws, int) and shard_ws > 1:
                shards_ready = (cache_dir / _READY_FILE).is_file() and all(
                    (cache_dir / f"shard_{r}.json").is_file() for r in range(shard_ws)
                )
            staged = {**staged, "on_disk": on_disk, "shards_ready": shards_ready, **paths}
    return {
        "collection_id": collection_id,
        "name": meta.get("name") or collection_id,
        "description": meta.get("description"),
        "local_path": str(root),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "last_staged": staged,
        "paths": paths or {"local_path": str(root)},
    }


def stage_data_collection(
    *,
    collection_id: str,
    dataset: TrainingDatasetConfig | dict[str, Any],
    task_type: str,
    ray_train_workers: int | None = None,
) -> dict[str, Any]:
    root = collection_dir(collection_id)
    meta = _read_collection_meta(root)
    if meta is None:
        raise ValueError(f"Data collection not found: {collection_id}")

    dataset_cfg = dataset.model_dump() if isinstance(dataset, TrainingDatasetConfig) else dict(dataset)
    staging_root = str(root)
    cache_key = _cache_key(dataset_cfg, task_type)
    paths = local_path_fields(staging_root, dataset_cfg, task_type)
    now = datetime.now(timezone.utc).isoformat()

    if is_remote_dataset(dataset_cfg):
        remote_uri = remote_data_uri(dataset_cfg)
        last_staged = {
            "cache_key": cache_key,
            "task_type": task_type,
            "access_mode": "remote",
            "remote_uri": remote_uri,
            "dataset_summary": _dataset_summary(dataset_cfg),
            "corpus_examples": None,
            "shard_world_size": None,
            "staged_at": now,
            "local_path": paths["local_path"],
            "corpus_local_path": None,
            "cache_local_path": paths["cache_local_path"],
        }
        meta["last_staged"] = last_staged
        meta["updated_at"] = now
        _write_collection_meta(collection_id, meta)
        logger.info(
            "Registered remote data collection=%s remote_uri=%s registry_local_path=%s",
            collection_id,
            remote_uri,
            paths["local_path"],
        )
        return {
            "collection_id": collection_id,
            "name": meta.get("name") or collection_id,
            "access_mode": "remote",
            "remote_uri": remote_uri,
            "local_path": paths["local_path"],
            "corpus_local_path": None,
            "cache_local_path": paths["cache_local_path"],
            "corpus_examples": None,
            "task_type": task_type,
            "shard_meta": None,
            "last_staged": last_staged,
        }

    texts, _labels = load_training_corpus_staged(
        artifact_dir=staging_root,
        dataset_cfg=dataset_cfg,
        task_type=task_type,
    )
    paths = local_path_fields(staging_root, dataset_cfg, task_type)

    shard_meta: dict[str, Any] | None = None
    if ray_train_workers is not None and ray_train_workers > 1:
        shard_meta = prepare_dataset_shards_on_driver(
            artifact_dir=staging_root,
            dataset_cfg=dataset_cfg,
            task_type=task_type,
            world_size=ray_train_workers,
        )

    last_staged = {
        "cache_key": cache_key,
        "task_type": task_type,
        "access_mode": "local",
        "remote_uri": None,
        "dataset_summary": _dataset_summary(dataset_cfg),
        "corpus_examples": len(texts),
        "shard_world_size": int(shard_meta["world_size"]) if shard_meta else None,
        "staged_at": now,
        **paths,
    }
    meta["last_staged"] = last_staged
    meta["updated_at"] = now
    _write_collection_meta(collection_id, meta)

    logger.info(
        "Staged data collection=%s examples=%s local_path=%s corpus_local_path=%s",
        collection_id,
        len(texts),
        paths["local_path"],
        paths["corpus_local_path"],
    )
    return {
        "collection_id": collection_id,
        "name": meta.get("name") or collection_id,
        "access_mode": "local",
        "remote_uri": None,
        **paths,
        "corpus_examples": len(texts),
        "task_type": task_type,
        "shard_meta": shard_meta,
        "last_staged": last_staged,
    }


def _dataset_summary(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    source = str(dataset_cfg.get("source") or "")
    summary: dict[str, Any] = {"source": source}
    if source == "huggingface":
        summary["hf_dataset"] = dataset_cfg.get("hf_dataset")
        summary["hf_split"] = dataset_cfg.get("hf_split")
    elif source in ("text_file", "jsonl_file"):
        summary["path"] = dataset_cfg.get("path")
    elif source == "builtin":
        summary["builtin_name"] = dataset_cfg.get("builtin_name")
    elif source in REMOTE_DATASET_SOURCES:
        summary["remote_uri"] = remote_data_uri(dataset_cfg)
    return summary


def find_collection_for_dataset(
    dataset_cfg: dict[str, Any],
    task_type: str,
    *,
    preferred_collection_id: str | None = None,
) -> dict[str, Any] | None:
    """Locate a shared collection that already has this dataset staged on disk."""
    cache_key = _cache_key(dataset_cfg, task_type)

    def _match(collection_id: str) -> dict[str, Any] | None:
        root = COLLECTIONS_ROOT / collection_id
        if not root.is_dir():
            return None
        meta = _read_collection_meta(root)
        staged = (meta or {}).get("last_staged")
        if isinstance(staged, dict) and staged.get("cache_key") == cache_key:
            if staged.get("access_mode") == "remote" or is_remote_dataset(dataset_cfg):
                return {
                    "collection_id": collection_id,
                    "name": (meta or {}).get("name") or collection_id,
                    "local_path": str(root.resolve()),
                    "access_mode": "remote",
                    "remote_uri": staged.get("remote_uri") or remote_data_uri(dataset_cfg),
                    "cache_key": cache_key,
                }
        cache = root / ".dataset_shards" / cache_key
        corpus = cache / _CORPUS_FILE
        if not corpus.is_file():
            return None
        paths = local_path_fields(root, dataset_cfg, task_type)
        return {
            "collection_id": collection_id,
            "name": (meta or {}).get("name") or collection_id,
            "access_mode": "local",
            **paths,
            "cache_key": cache_key,
        }

    if preferred_collection_id:
        hit = _match(preferred_collection_id)
        if hit:
            return hit

    for row in list_data_collections():
        hit = _match(str(row["collection_id"]))
        if hit:
            return hit
    return None


def shared_dataset_dirs_from_config(train_config: dict[str, Any]) -> list[str] | None:
    raw = train_config.get("shared_dataset_dirs")
    if isinstance(raw, list):
        dirs = [str(d).strip() for d in raw if str(d).strip()]
        if dirs:
            return dirs
    path = train_config.get("data_collection_local_path")
    if isinstance(path, str) and path.strip():
        return [path.strip()]
    return None


def attach_shared_collection_to_train_config(
    train_config: dict[str, Any],
    *,
    dataset_cfg: dict[str, Any],
    task_type: str,
) -> dict[str, Any]:
    """
    Point train_config at a shared collection directory when corpus is already staged.

    Workers and the driver read from ``shared_dataset_dirs`` (collection local path)
    before copying or reloading into the per-job artifact folder.
    """
    preferred = train_config.get("data_collection_id")
    if isinstance(preferred, str):
        preferred = preferred.strip() or None
    hit = find_collection_for_dataset(
        dataset_cfg,
        task_type,
        preferred_collection_id=preferred,
    )
    if not hit:
        return train_config

    train_config["data_collection_id"] = hit["collection_id"]
    train_config["data_collection_local_path"] = hit["local_path"]
    if hit.get("access_mode") == "remote" or is_remote_dataset(dataset_cfg):
        train_config["dataset_access_mode"] = "remote"
        uri = hit.get("remote_uri") or remote_data_uri(dataset_cfg)
        if uri:
            train_config["dataset_remote_uri"] = uri
        train_config.pop("shared_dataset_dirs", None)
        return train_config

    local_path = hit["local_path"]
    train_config["dataset_access_mode"] = "local"
    train_config["shared_dataset_dirs"] = [local_path]
    return train_config


def seed_job_artifact_from_shared_collection(
    *,
    job_artifact_dir: str,
    dataset_cfg: dict[str, Any],
    task_type: str,
    collection_local_path: str,
) -> dict[str, Any]:
    """Optional copy into job artifact when workers cannot see the collection path."""
    src_cache = shard_cache_dir(collection_local_path, dataset_cfg, task_type)
    if not (src_cache / _CORPUS_FILE).is_file():
        return {"copied": False, "reason": "no_corpus_on_collection", "from": str(src_cache)}

    dest_cache = shard_cache_dir(job_artifact_dir, dataset_cfg, task_type)
    if (dest_cache / _CORPUS_FILE).is_file():
        return {
            "copied": False,
            "reason": "job_already_staged",
            "to": str(dest_cache),
        }

    _copy_cache_tree(src_cache, dest_cache)
    return {
        "copied": True,
        "from": str(src_cache),
        "to": str(dest_cache),
        "corpus_local_path": str(dest_cache / _CORPUS_FILE),
    }


def _copy_cache_tree(src_cache: Path, dest_cache: Path) -> None:
    dest_cache.mkdir(parents=True, exist_ok=True)
    for item in src_cache.iterdir():
        if not item.is_file():
            continue
        target = dest_cache / item.name
        if target.is_file():
            continue
        shutil.copy2(item, target)


# —— Legacy project paths (same layout as collections, under project_{id}) ——

def project_staging_dir(project_id: int) -> Path:
    root = (STAGING_ROOT / f"project_{project_id}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
