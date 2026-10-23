"""Shard training text on the driver so Ray Train workers avoid full-dataset tokenization."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from app.training.dataset_loader import load_training_samples
from app.training.dataset_config import is_remote_dataset

logger = logging.getLogger(__name__)

_READY_FILE = "ready.json"
_CORPUS_FILE = "corpus.json"
_SHARD_WAIT_SEC = 900.0
_SHARD_POLL_SEC = 0.25


def _cache_key(dataset_cfg: dict[str, Any], task_type: str) -> str:
    payload = json.dumps({"dataset": dataset_cfg, "task_type": task_type}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def shard_cache_dir(artifact_dir: str | Path, dataset_cfg: dict[str, Any], task_type: str) -> Path:
    base = Path(artifact_dir).resolve() / ".dataset_shards" / _cache_key(dataset_cfg, task_type)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_shard(path: Path, texts: list[str], labels: list[int] | None) -> None:
    payload: dict[str, Any] = {"texts": texts}
    if labels is not None:
        payload["labels"] = labels
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_corpus(path: Path) -> tuple[list[str], list[int] | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Staged corpus missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid staged corpus: {path}")
    texts = data.get("texts")
    if not isinstance(texts, list):
        raise ValueError(f"Staged corpus texts missing in {path}")
    labels_raw = data.get("labels")
    labels: list[int] | None = None
    if labels_raw is not None:
        if not isinstance(labels_raw, list):
            raise ValueError(f"Staged corpus labels invalid in {path}")
        labels = [int(x) for x in labels_raw]
    return [str(t) for t in texts], labels


def _write_corpus(path: Path, texts: list[str], labels: list[int] | None) -> None:
    payload: dict[str, Any] = {"texts": texts}
    if labels is not None:
        payload["labels"] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _shared_staging_roots(
    artifact_dir: str | None,
    shared_dirs: Sequence[str] | None,
) -> list[str]:
    roots: list[str] = []
    for raw in shared_dirs or ():
        path = str(raw or "").strip()
        if path and path not in roots:
            roots.append(path)
    if artifact_dir:
        path = str(artifact_dir).strip()
        if path and path not in roots:
            roots.append(path)
    return roots


def load_training_corpus_staged(
    *,
    artifact_dir: str | None,
    dataset_cfg: dict[str, Any],
    task_type: str,
    shared_dirs: Sequence[str] | None = None,
) -> tuple[list[str], list[int] | None]:
    """
    Load training texts once per job artifact + dataset config.

    Reuses ``.dataset_shards/<key>/corpus.json`` when present so Hugging Face /
    file loads are not repeated across prompt-format detection, tokenizer prep,
    and driver sharding.

    Iceberg and S3 Parquet are always read from remote at use time (no local corpus copy).
    """
    if is_remote_dataset(dataset_cfg):
        return load_training_samples(dataset_cfg, task_type=task_type)

    for root in _shared_staging_roots(artifact_dir, shared_dirs):
        cache = shard_cache_dir(root, dataset_cfg, task_type)
        corpus_path = cache / _CORPUS_FILE
        if corpus_path.is_file():
            try:
                texts, labels = _read_corpus(corpus_path)
                logger.info(
                    "Reusing staged training corpus examples=%s path=%s",
                    len(texts),
                    corpus_path,
                )
                return texts, labels
            except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
                logger.warning("Ignoring invalid staged corpus %s: %s", corpus_path, exc)

    source = str(dataset_cfg.get("source") or "unknown")
    logger.info("Dataset download/load — started (source=%s)", source)
    t0 = time.perf_counter()
    texts, labels = load_training_samples(dataset_cfg, task_type=task_type)
    logger.info(
        "Dataset download/load — done (%s samples, %.0fs)",
        len(texts),
        time.perf_counter() - t0,
    )
    write_root = artifact_dir or (shared_dirs[0] if shared_dirs else None)
    if write_root:
        try:
            _write_corpus(
                shard_cache_dir(write_root, dataset_cfg, task_type) / _CORPUS_FILE,
                texts,
                labels,
            )
        except OSError as exc:
            logger.warning("Could not write staged corpus under %s: %s", write_root, exc)
    return texts, labels


def _prepared_shards_in_root(
    root: str,
    dataset_cfg: dict[str, Any],
    task_type: str,
    world_size: int,
) -> dict[str, Any] | None:
    cache = shard_cache_dir(root, dataset_cfg, task_type)
    ready_path = cache / _READY_FILE
    if not ready_path.is_file():
        return None
    try:
        meta = json.loads(ready_path.read_text(encoding="utf-8"))
        if int(meta.get("world_size", 0)) != world_size:
            return None
    except (json.JSONDecodeError, TypeError, ValueError, OSError):
        return None

    for rank in range(world_size):
        if not (cache / f"shard_{rank}.json").is_file():
            return None
    return {**meta, "cache_dir": str(cache), "staging_root": str(Path(root).resolve())}


def prepared_dataset_shards_meta(
    *,
    artifact_dir: str,
    dataset_cfg: dict[str, Any],
    task_type: str,
    world_size: int,
    shared_dirs: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Return shard metadata when ready.json and all shard files exist on disk."""
    if world_size <= 1:
        return {"world_size": 1, "skipped": True}

    for root in _shared_staging_roots(artifact_dir, shared_dirs):
        meta = _prepared_shards_in_root(root, dataset_cfg, task_type, world_size)
        if meta is not None:
            return meta
    return None


def _read_shard(path: Path) -> tuple[list[str], list[int] | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset shard missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid shard file: {path}")
    texts = data.get("texts")
    if not isinstance(texts, list):
        raise ValueError(f"Shard texts missing in {path}")
    labels_raw = data.get("labels")
    labels: list[int] | None = None
    if labels_raw is not None:
        if not isinstance(labels_raw, list):
            raise ValueError(f"Shard labels invalid in {path}")
        labels = [int(x) for x in labels_raw]
    return [str(t) for t in texts], labels


def split_interleaved(
    texts: list[str], labels: list[int] | None, world_size: int
) -> list[tuple[list[str], list[int] | None]]:
    shards: list[tuple[list[str], list[int] | None]] = []
    for rank in range(world_size):
        shard_texts = texts[rank::world_size]
        shard_labels = labels[rank::world_size] if labels is not None else None
        shards.append((shard_texts, shard_labels))
    return shards


def prepare_dataset_shards_on_driver(
    *,
    artifact_dir: str,
    dataset_cfg: dict[str, Any],
    task_type: str,
    world_size: int,
    shared_dirs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Load the corpus once on the Train driver and write one JSON shard per worker.

    Workers read only their shard (under the shared job artifact_dir). The full Hugging
    Face / file load happens once; tokenization still runs per worker on a 1/world_size slice.
    """
    if world_size <= 1:
        return {"world_size": 1, "skipped": True}

    existing = prepared_dataset_shards_meta(
        artifact_dir=artifact_dir,
        dataset_cfg=dataset_cfg,
        task_type=task_type,
        world_size=world_size,
        shared_dirs=shared_dirs,
    )
    if existing is not None:
        logger.info(
            "Reusing dataset shards on disk world_size=%s total=%s root=%s",
            world_size,
            existing.get("total_examples"),
            existing.get("staging_root") or artifact_dir,
        )
        return existing

    write_root = artifact_dir
    if shared_dirs:
        for root in shared_dirs:
            cache = shard_cache_dir(root, dataset_cfg, task_type)
            if (cache / _CORPUS_FILE).is_file():
                write_root = root
                break

    cache = shard_cache_dir(write_root, dataset_cfg, task_type)
    ready_path = cache / _READY_FILE

    texts, labels = load_training_corpus_staged(
        artifact_dir=artifact_dir,
        dataset_cfg=dataset_cfg,
        task_type=task_type,
        shared_dirs=shared_dirs,
    )
    shards = split_interleaved(texts, labels, world_size)
    for rank, (shard_texts, shard_labels) in enumerate(shards):
        _write_shard(cache / f"shard_{rank}.json", shard_texts, shard_labels)

    meta = {
        "world_size": world_size,
        "total_examples": len(texts),
        "examples_per_shard": [len(s[0]) for s in shards],
        "cache_dir": str(cache),
    }
    ready_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Prepared dataset shards total=%s world_size=%s per_shard=%s",
        len(texts),
        world_size,
        meta["examples_per_shard"],
    )
    return meta


def _resolve_shard_cache_dir(
    artifact_dir: str,
    dataset_cfg: dict[str, Any],
    task_type: str,
    shared_dirs: Sequence[str] | None,
) -> Path:
    for root in _shared_staging_roots(artifact_dir, shared_dirs):
        cache = shard_cache_dir(root, dataset_cfg, task_type)
        if (cache / _READY_FILE).is_file():
            return cache
    return shard_cache_dir(artifact_dir, dataset_cfg, task_type)


def load_dataset_shard_for_worker(
    *,
    artifact_dir: str,
    dataset_cfg: dict[str, Any],
    task_type: str,
    rank: int,
    world_size: int,
    shared_dirs: Sequence[str] | None = None,
) -> tuple[list[str], list[int] | None]:
    """Load this Train rank's text shard (wait for driver-prepared cache if needed)."""
    if world_size <= 1:
        return load_training_corpus_staged(
            artifact_dir=artifact_dir,
            dataset_cfg=dataset_cfg,
            task_type=task_type,
            shared_dirs=shared_dirs,
        )

    cache = _resolve_shard_cache_dir(artifact_dir, dataset_cfg, task_type, shared_dirs)
    ready_path = cache / _READY_FILE
    shard_path = cache / f"shard_{rank}.json"
    deadline = time.monotonic() + _SHARD_WAIT_SEC
    while not shard_path.is_file() or not ready_path.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for dataset shard rank={rank} under {cache}. "
                "Ensure job artifact_dir is on shared project storage visible to CDSW workers."
            )
        time.sleep(_SHARD_POLL_SEC)

    texts, labels = _read_shard(shard_path)
    logger.info(
        "Worker rank=%s loaded shard examples=%s (world_size=%s)",
        rank,
        len(texts),
        world_size,
    )
    return texts, labels
