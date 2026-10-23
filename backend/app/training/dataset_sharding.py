"""Shard training text on the driver so Ray Train workers avoid full-dataset tokenization."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from app.services.download_progress import DownloadProgressState
from app.training.corpus_staging import (
    CORPUS_JSONL,
    CORPUS_META,
    PROMPT_FORMAT_PROBE_ROWS,
    StagedCorpusRef,
    iter_corpus_jsonl,
    iter_corpus_jsonl_sample,
    iter_shard_jsonl,
    read_corpus_meta,
)
from app.training.dataset_config import is_remote_dataset
from app.training.dataset_loader import (
    ProgressCallback,
    load_training_samples,
    should_stage_corpus_on_disk,
    stream_dataset_to_corpus_jsonl,
)

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


def resolve_staged_corpus_jsonl(
    *,
    artifact_dir: str | None,
    dataset_cfg: dict[str, Any],
    task_type: str,
    shared_dirs: Sequence[str] | None = None,
) -> Path | None:
    for root in _shared_staging_roots(artifact_dir, shared_dirs):
        cache = shard_cache_dir(root, dataset_cfg, task_type)
        jsonl = cache / CORPUS_JSONL
        meta = cache / CORPUS_META
        if jsonl.is_file() and meta.is_file() and read_corpus_meta(meta):
            return jsonl
    return None


def staged_corpus_ref(
    *,
    artifact_dir: str | None,
    dataset_cfg: dict[str, Any],
    task_type: str,
    shared_dirs: Sequence[str] | None = None,
) -> StagedCorpusRef | None:
    for root in _shared_staging_roots(artifact_dir, shared_dirs):
        cache = shard_cache_dir(root, dataset_cfg, task_type)
        jsonl = cache / CORPUS_JSONL
        meta_path = cache / CORPUS_META
        if jsonl.is_file() and meta_path.is_file():
            meta = read_corpus_meta(meta_path)
            if meta:
                count = int(meta.get("total_examples") or 0)
                return StagedCorpusRef(
                    texts=[],
                    labels=None,
                    jsonl_path=jsonl,
                    example_count=count,
                    on_disk_only=True,
                )
        corpus_path = cache / _CORPUS_FILE
        if corpus_path.is_file():
            try:
                texts, labels = _read_corpus(corpus_path)
                return StagedCorpusRef(
                    texts=texts,
                    labels=labels,
                    jsonl_path=None,
                    example_count=len(texts),
                    on_disk_only=False,
                )
            except (json.JSONDecodeError, TypeError, ValueError, OSError):
                continue
    return None


def load_prompt_format_probe_sample(
    dataset_cfg: dict[str, Any],
    *,
    task_type: str,
    artifact_dir: str | None = None,
    shared_dirs: Sequence[str] | None = None,
) -> list[str]:
    """Never load full multi-million-row corpora for ``prompt_format=auto``."""
    jsonl = resolve_staged_corpus_jsonl(
        artifact_dir=artifact_dir,
        dataset_cfg=dataset_cfg,
        task_type=task_type,
        shared_dirs=shared_dirs,
    )
    if jsonl is not None:
        return iter_corpus_jsonl_sample(jsonl, PROMPT_FORMAT_PROBE_ROWS)

    from app.training.dataset_config import TrainingDatasetConfig

    cfg = TrainingDatasetConfig.model_validate(dataset_cfg)
    probe = cfg.model_copy(deep=True)
    if probe.max_samples is None or probe.max_samples > PROMPT_FORMAT_PROBE_ROWS:
        probe.max_samples = PROMPT_FORMAT_PROBE_ROWS
    texts, _ = load_training_samples(probe, task_type=task_type)
    return texts


def load_training_corpus_staged(
    *,
    artifact_dir: str | None,
    dataset_cfg: dict[str, Any],
    task_type: str,
    shared_dirs: Sequence[str] | None = None,
    download_progress: DownloadProgressState | None = None,
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

    existing = staged_corpus_ref(
        artifact_dir=artifact_dir,
        dataset_cfg=dataset_cfg,
        task_type=task_type,
        shared_dirs=shared_dirs,
    )
    if existing is not None:
        if existing.on_disk_only:
            logger.info(
                "Reusing disk-staged corpus examples=%s path=%s",
                existing.example_count,
                existing.jsonl_path,
            )
            return [], None
        logger.info(
            "Reusing staged training corpus examples=%s",
            existing.example_count,
        )
        return existing.texts, existing.labels

    source = str(dataset_cfg.get("source") or "unknown")
    logger.info("Dataset download/load — started (source=%s)", source)
    t0 = time.perf_counter()

    progress_cb: ProgressCallback | None = None
    if download_progress is not None:

        def progress_cb(count: int, limit: int | None) -> None:
            download_progress.update_rows(count, limit)

    write_root = artifact_dir or (shared_dirs[0] if shared_dirs else None)
    cache = shard_cache_dir(write_root, dataset_cfg, task_type) if write_root else None

    if should_stage_corpus_on_disk(dataset_cfg, task_type=task_type):
        if not cache:
            raise ValueError("artifact_dir required to stage large datasets on disk")
        jsonl_path = cache / CORPUS_JSONL
        meta_path = cache / CORPUS_META
        logger.info(
            "Staging large dataset to disk (session RAM cap avoided): %s",
            jsonl_path,
        )
        count = stream_dataset_to_corpus_jsonl(
            dataset_cfg,
            task_type=task_type,
            jsonl_path=jsonl_path,
            meta_path=meta_path,
            progress_cb=progress_cb,
        )
        logger.info(
            "Dataset download/load — done (%s examples on disk, %.0fs)",
            count,
            time.perf_counter() - t0,
        )
        return [], None

    texts, labels = load_training_samples(
        dataset_cfg,
        task_type=task_type,
        progress_cb=progress_cb,
    )
    logger.info(
        "Dataset download/load — done (%s samples, %.0fs)",
        len(texts),
        time.perf_counter() - t0,
    )
    if write_root and cache:
        try:
            _write_corpus(cache / _CORPUS_FILE, texts, labels)
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
        json_shard = cache / f"shard_{rank}.json"
        jsonl_shard = cache / f"shard_{rank}.jsonl"
        if not json_shard.is_file() and not jsonl_shard.is_file():
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
    jsonl_path = path.with_suffix(".jsonl")
    if jsonl_path.is_file():
        texts = list(iter_shard_jsonl(jsonl_path))
        return texts, None
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


def _write_shards_from_corpus_jsonl(
    corpus_jsonl: Path,
    cache: Path,
    world_size: int,
) -> tuple[int, list[int]]:
    """Stream interleaved shard JSONL files without holding the full corpus in RAM."""
    handles: list[Any] = []
    for rank in range(world_size):
        shard_path = cache / f"shard_{rank}.jsonl"
        if shard_path.is_file():
            shard_path.unlink()
        handles.append(shard_path.open("w", encoding="utf-8"))
    counts = [0] * world_size
    total = 0
    try:
        for idx, text in enumerate(iter_corpus_jsonl(corpus_jsonl)):
            rank = idx % world_size
            handles[rank].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            counts[rank] += 1
            total += 1
    finally:
        for fp in handles:
            fp.close()
    return total, counts


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
    download_progress: DownloadProgressState | None = None,
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
            if (cache / _CORPUS_FILE).is_file() or (cache / CORPUS_JSONL).is_file():
                write_root = root
                break

    cache = shard_cache_dir(write_root, dataset_cfg, task_type)
    ready_path = cache / _READY_FILE

    texts, labels = load_training_corpus_staged(
        artifact_dir=artifact_dir,
        dataset_cfg=dataset_cfg,
        task_type=task_type,
        shared_dirs=shared_dirs,
        download_progress=download_progress,
    )
    corpus_jsonl = cache / CORPUS_JSONL
    if corpus_jsonl.is_file() and not texts:
        total, per_shard = _write_shards_from_corpus_jsonl(corpus_jsonl, cache, world_size)
        meta = {
            "world_size": world_size,
            "total_examples": total,
            "examples_per_shard": per_shard,
            "cache_dir": str(cache),
            "shard_format": "jsonl",
        }
    else:
        shards = split_interleaved(texts, labels, world_size)
        for rank, (shard_texts, shard_labels) in enumerate(shards):
            _write_shard(cache / f"shard_{rank}.json", shard_texts, shard_labels)
        meta = {
            "world_size": world_size,
            "total_examples": len(texts),
            "examples_per_shard": [len(s[0]) for s in shards],
            "cache_dir": str(cache),
            "shard_format": "json",
        }
    ready_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Prepared dataset shards total=%s world_size=%s per_shard=%s",
        meta["total_examples"],
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
        ref = staged_corpus_ref(
            artifact_dir=artifact_dir,
            dataset_cfg=dataset_cfg,
            task_type=task_type,
            shared_dirs=shared_dirs,
        )
        if ref is not None and ref.on_disk_only and ref.jsonl_path is not None:
            from app.training.corpus_staging import corpus_stage_mode

            if corpus_stage_mode() == "memory":
                texts = list(iter_corpus_jsonl(ref.jsonl_path))
                return texts, None
            raise ValueError(
                "Corpus is staged on disk under the job artifact (.dataset_shards/…/corpus.jsonl). "
                "Use multiple Ray Train workers so shards load per rank, or set "
                "PLATFORM_CORPUS_STAGE_MODE=memory to load the full corpus into RAM for single-worker runs."
            )
        return load_training_corpus_staged(
            artifact_dir=artifact_dir,
            dataset_cfg=dataset_cfg,
            task_type=task_type,
            shared_dirs=shared_dirs,
        )

    cache = _resolve_shard_cache_dir(artifact_dir, dataset_cfg, task_type, shared_dirs)
    ready_path = cache / _READY_FILE
    shard_json = cache / f"shard_{rank}.json"
    shard_jsonl = cache / f"shard_{rank}.jsonl"
    deadline = time.monotonic() + _SHARD_WAIT_SEC
    while (not shard_json.is_file() and not shard_jsonl.is_file()) or not ready_path.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for dataset shard rank={rank} under {cache}. "
                "Ensure job artifact_dir is on shared project storage visible to CDSW workers."
            )
        time.sleep(_SHARD_POLL_SEC)

    texts, labels = _read_shard(shard_json)
    logger.info(
        "Worker rank=%s loaded shard examples=%s (world_size=%s)",
        rank,
        len(texts),
        world_size,
    )
    return texts, labels
