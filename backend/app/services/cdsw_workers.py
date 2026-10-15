"""
Cloudera AI / CDSW workers integration for Ray GPU clusters.

Uses the workers API documented at:
https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html

Supports both legacy `cdsw` and `cml.workers_v1` entry points when running inside
Cloudera Machine Learning. Outside CML, operations are simulated for local development.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class WorkersClient(Protocol):
    def launch_workers(
        self,
        n: int,
        cpu: float,
        memory: float,
        *,
        nvidia_gpu: int = 0,
        code: str = "",
        env: dict[str, str] | None = None,
        accelerator_label_id: int | None = None,
        resource_profile_id: int | None = None,
        resource_group_id: int | None = None,
    ) -> Any: ...

    def stop_workers(self, *worker_ids: int) -> None: ...

    def list_workers(self) -> list[dict[str, Any]]: ...


@dataclass
class LaunchedWorkers:
    worker_ids: list[str]
    raw: Any


def _import_workers_client() -> WorkersClient | None:
    try:
        import cml.workers_v1 as workers  # type: ignore[import-untyped]

        return workers
    except ImportError:
        pass
    try:
        import cdsw  # type: ignore[import-untyped]

        return cdsw
    except ImportError:
        return None


def cdsw_available() -> bool:
    return _import_workers_client() is not None


def ray_head_address() -> str | None:
    """Best-effort Ray head address when running inside a CML/CDSW session."""
    ip = os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")
    if ip:
        return f"{ip}:6379"
    return os.environ.get("RAY_HEAD_ADDRESS")


def ray_client_address(head: str) -> str:
    host = head.split(":")[0]
    return f"ray://{host}:10001"


def build_worker_start_code(head_address: str) -> str:
    """Python code passed to launch_workers `code=` to join the Ray cluster."""
    # Worker engines often lack `ray` on PATH; use the session Python module entrypoint.
    return (
        "import subprocess, sys\n"
        "cmd = [sys.executable, '-m', 'ray.scripts.scripts', 'start', "
        f"'--block', '--address={head_address}']\n"
        "sys.exit(subprocess.call(cmd))\n"
    )


class CdswRayWorkerPool:
    """Provision GPU workers via cdsw/cml workers API and attach them to Ray."""

    def __init__(
        self,
        cpu: float = 4.0,
        memory_gb: float = 16.0,
        gpus_per_worker: int = 1,
    ) -> None:
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.gpus_per_worker = gpus_per_worker
        self._client = _import_workers_client()
        self._launched: LaunchedWorkers | None = None

    @property
    def is_remote(self) -> bool:
        return self._client is not None

    def launch(self, count: int, head_address: str) -> LaunchedWorkers:
        if count < 1:
            raise ValueError("count must be >= 1")
        code = build_worker_start_code(head_address)
        if self._client is None:
            simulated = [f"local-sim-{i}" for i in range(count)]
            logger.info(
                "CDSW/CML workers API not available; simulating %s workers", count
            )
            self._launched = LaunchedWorkers(worker_ids=simulated, raw=simulated)
            return self._launched

        kwargs: dict[str, Any] = {
            "n": count,
            "cpu": self.cpu,
            "memory": self.memory_gb,
            "nvidia_gpu": self.gpus_per_worker,
            "code": code,
            "env": {},
        }
        optional = {
            "accelerator_label_id": os.environ.get("CML_ACCELERATOR_LABEL_ID"),
            "resource_profile_id": os.environ.get("CML_RESOURCE_PROFILE_ID"),
            "resource_group_id": os.environ.get("CML_RESOURCE_GROUP_ID"),
        }
        for key, val in optional.items():
            if val is not None:
                kwargs[key] = int(val)

        raw = self._client.launch_workers(**kwargs)
        worker_ids = _extract_worker_ids(raw)
        self._launched = LaunchedWorkers(worker_ids=worker_ids, raw=raw)
        logger.info("Launched %s CDSW workers for Ray head %s", count, head_address)
        return self._launched

    def scale(self, target: int, head_address: str) -> LaunchedWorkers:
        current = len(self._launched.worker_ids) if self._launched else 0
        if target > current:
            delta = target - current
            extra = self.launch(delta, head_address)
            merged = (self._launched.worker_ids if self._launched else []) + extra.worker_ids
            self._launched = LaunchedWorkers(worker_ids=merged, raw=extra.raw)
        elif target < current and self._launched:
            to_stop = self._launched.worker_ids[target:]
            self.stop_ids(to_stop)
            self._launched = LaunchedWorkers(
                worker_ids=self._launched.worker_ids[:target], raw=self._launched.raw
            )
        return self._launched or LaunchedWorkers(worker_ids=[], raw=[])

    def stop_ids(self, worker_ids: list[str]) -> None:
        if not worker_ids or self._client is None:
            return
        numeric = [int(w) for w in worker_ids if str(w).isdigit()]
        if numeric:
            self._client.stop_workers(*numeric)

    def stop_all(self) -> None:
        if self._launched:
            self.stop_ids(self._launched.worker_ids)
            self._launched = None
        if self._client is not None:
            try:
                self._client.stop_workers()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop all workers")


def _extract_worker_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        ids: list[str] = []
        for item in raw:
            if isinstance(item, dict) and "id" in item:
                ids.append(str(item["id"]))
            else:
                ids.append(str(item))
        return ids
    if isinstance(raw, dict) and "id" in raw:
        return [str(raw["id"])]
    return [str(raw)]


def worker_ids_to_json(ids: list[str]) -> str:
    return json.dumps(ids)
