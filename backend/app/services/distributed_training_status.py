"""Live distributed training indicators (Ray Train ranks, devices, world size)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT

_RANK_LINE = re.compile(
    r"rank\s+(\d+)/(\d+)\s+ready\s+device=([^\s]+)",
    re.IGNORECASE,
)


def distributed_status_path(job_id: int) -> Path:
    return REPO_ROOT / "data" / "models" / f"job_{job_id}" / "distributed.json"


def update_distributed_rank(
    artifact_dir: str,
    *,
    rank: int,
    world_size: int,
    device: str,
    host: str,
    collective_backend: str | None = None,
) -> None:
    """Called from each Ray Train worker when the train loop starts."""
    path = Path(artifact_dir) / "distributed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    data: dict[str, Any] = {
        "backend": "ray.train.torch",
        "world_size": world_size,
        "ranks": [],
    }
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    ranks_map: dict[int, dict[str, Any]] = {}
    for entry in data.get("ranks") or []:
        if isinstance(entry, dict) and "rank" in entry:
            ranks_map[int(entry["rank"])] = entry
    ranks_map[rank] = {
        "rank": rank,
        "display_rank": rank + 1,
        "world_size": world_size,
        "device": device,
        "host": host,
        "last_seen": now,
        "state": "training",
        "collective_backend": collective_backend,
    }
    data["backend"] = "ray.train.torch"
    data["world_size"] = world_size
    data["ranks"] = sorted(ranks_map.values(), key=lambda r: int(r["rank"]))
    data["updated_at"] = now
    data["multi_worker"] = world_size > 1
    if collective_backend:
        data["collective_backend"] = collective_backend
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_ranks_from_training_log(lines: list[dict[str, str]]) -> list[dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for line in lines:
        msg = line.get("message") or ""
        m = _RANK_LINE.search(msg)
        if not m:
            continue
        display_rank = int(m.group(1))
        world_size = int(m.group(2))
        rank = display_rank - 1
        found[rank] = {
            "rank": rank,
            "display_rank": display_rank,
            "world_size": world_size,
            "device": m.group(3),
            "host": "",
            "last_seen": line.get("ts"),
            "state": "training",
            "source": "training_log",
        }
    return sorted(found.values(), key=lambda r: r["rank"])


def build_distributed_training_status(
    *,
    job_id: int,
    job_status: str,
    num_cdsw_workers: int,
    ray_train_workers: int,
    gpus_per_worker: int,
    training_log: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Truthful distributed-training snapshot for the UI.

    `active` means at least one Ray Train worker has entered the train loop.
    `multi_node` means world_size > 1 or multiple alive ranks reported.
    """
    file_data: dict[str, Any] = {}
    path = distributed_status_path(job_id)
    if path.is_file():
        try:
            file_data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            file_data = {}

    ranks = file_data.get("ranks") if isinstance(file_data.get("ranks"), list) else []
    if not ranks:
        ranks = _parse_ranks_from_training_log(training_log)

    world_size = int(file_data.get("world_size") or 0)
    if not world_size and ranks:
        world_size = int(ranks[0].get("world_size") or len(ranks))

    configured_train = max(1, int(ray_train_workers or 1))
    cdsw_engines = max(0, int(num_cdsw_workers or 0))
    ranks_live = len(ranks)
    multi_train_ranks = world_size > 1 or configured_train > 1
    multi_engine_cluster = cdsw_engines > 1
    # Legacy flag: true only when TorchTrainer runs multiple ranks (not CDSW engine count).
    multi_worker = multi_train_ranks
    all_ranks_reported = world_size > 0 and ranks_live >= world_size

    running = job_status in ("provisioning", "running", "queued")
    active = ranks_live > 0 and job_status in ("running", "succeeded")
    succeeded = job_status == "succeeded"

    if succeeded and ranks_live == 0 and configured_train >= 1:
        active = True
        if not world_size:
            world_size = configured_train

    status_label = "idle"
    if succeeded:
        status_label = "completed"
    elif active and all_ranks_reported:
        status_label = "distributed_active"
    elif active:
        status_label = "partial"
    elif running:
        status_label = "starting"

    collective = file_data.get("collective_backend")
    if not collective and ranks:
        collective = ranks[0].get("collective_backend")
    if not collective and world_size > 1:
        collective = "nccl (expected for CUDA multi-rank)"

    return {
        "backend": file_data.get("backend") or "ray.train.torch",
        "collective_backend": collective,
        "nccl_defaults_enabled": __import__(
            "app.services.nccl_env", fromlist=["nccl_defaults_enabled"]
        ).nccl_defaults_enabled(),
        "status": status_label,
        "active": active,
        "multi_worker": multi_worker,
        "multi_train_ranks": multi_train_ranks,
        "multi_engine_cluster": multi_engine_cluster,
        "multi_node": multi_train_ranks and (cdsw_engines > 0 or world_size > 1),
        "configured": {
            "cdsw_worker_engines": cdsw_engines,
            "ray_train_workers": configured_train,
            "gpus_per_worker": gpus_per_worker,
        },
        "observed": {
            "world_size": world_size,
            "ranks_live": ranks_live,
            "all_ranks_reported": all_ranks_reported,
        },
        "ranks": ranks,
        "summary": _summary_text(
            status_label=status_label,
            world_size=world_size,
            ranks_live=ranks_live,
            configured_train=configured_train,
            num_cdsw_workers=cdsw_engines,
            multi_train_ranks=multi_train_ranks,
            multi_engine_cluster=multi_engine_cluster,
        ),
        "mode_label": _mode_label(
            configured_train=configured_train,
            world_size=world_size,
            cdsw_engines=cdsw_engines,
            multi_train_ranks=multi_train_ranks,
        ),
    }


def _mode_label(
    *,
    configured_train: int,
    world_size: int,
    cdsw_engines: int,
    multi_train_ranks: bool,
) -> str:
    train_n = world_size if world_size > 0 else configured_train
    if multi_train_ranks:
        return f"Distributed Train · {train_n} ranks"
    if cdsw_engines > 1:
        return f"{cdsw_engines} CDSW engines · 1 Train rank"
    return "Single Train rank"


def _summary_text(
    *,
    status_label: str,
    world_size: int,
    ranks_live: int,
    configured_train: int,
    num_cdsw_workers: int,
    multi_train_ranks: bool,
    multi_engine_cluster: bool,
) -> str:
    if status_label == "distributed_active":
        if multi_train_ranks:
            return (
                f"Distributed training is running: {ranks_live}/{world_size} Ray Train "
                f"worker ranks active (TorchTrainer, CDSW engines={num_cdsw_workers})."
            )
        if multi_engine_cluster:
            return (
                f"Training on 1 Ray Train rank across a {num_cdsw_workers}-engine Ray cluster "
                f"(CDSW workers provide GPUs; TorchTrainer world size {world_size or 1})."
            )
        return f"Training active on 1 Ray Train worker (world size {world_size or 1})."
    if status_label == "partial":
        return (
            f"Training starting: {ranks_live}/{max(world_size, configured_train)} "
            "worker ranks reported so far."
        )
    if status_label == "starting":
        return (
            f"Provisioning cluster for Ray Train ({configured_train} worker process"
            f"{'es' if configured_train != 1 else ''}, "
            f"{num_cdsw_workers} CDSW engine(s))."
        )
    if status_label == "completed":
        return "Training finished on the Ray Train worker group."
    return "Distributed training not started yet."
