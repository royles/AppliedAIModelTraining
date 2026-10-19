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

from app.config import REPO_ROOT
from app.resource_hints import log_worker_launch_memory

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


def _local_gcs_reachable(port: int = 6379) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def ray_driver_init_address(head: str | None) -> str | None:
    """
    Address for ray.init() from the CDSW session driver (same pod as the head).

    Remote worker engines must still use the session IP in build_worker_start_code.
    The driver must not dial the pod/container IP from inside the same pod (Ray GCS /
    observability pubsub time out). Platform settings may store the worker-facing
    head as 100.x.x.x while CDSW_IP_ADDRESS is unset in the API process.
    """
    if not head:
        env = os.environ.get("RAY_ADDRESS")
        if env and cdsw_available():
            return ray_driver_init_address(env) or env
        return env
    host, _, port = head.partition(":")
    port = port or "6379"
    try:
        port_i = int(port)
    except ValueError:
        return head

    loopback = f"127.0.0.1:{port_i}"
    if host in ("127.0.0.1", "localhost"):
        return loopback

    session_ip = os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")
    ray_head = os.environ.get("RAY_HEAD_ADDRESS") or ""
    ray_head_host = ray_head.partition(":")[0] if ray_head else ""

    # CAI/CML: training driver always runs in the session pod next to the Ray head.
    if cdsw_available():
        return loopback

    if session_ip and host == session_ip:
        return loopback
    if ray_head_host and host == ray_head_host:
        return loopback
    if _local_gcs_reachable(port_i):
        return loopback
    return head


def wait_for_gcs_port(
    host: str,
    port: int = 6379,
    *,
    timeout_sec: float = 180.0,
    poll_sec: float = 2.0,
) -> bool:
    """Block until Ray GCS accepts TCP connections (returns False on timeout)."""
    import socket
    import time

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(poll_sec)
    return False


def build_worker_start_code(head_address: str) -> str:
    """
    Python code for launch_workers `code=`.

    Always uses sys.executable (system Python). Installs backend/requirements.txt
    when Ray is missing (worker engines start with a bare interpreter).
    """
    # head_address is session-controlled (IP:port); no user strings in code path.
    return f'''import os
import subprocess
import sys
from pathlib import Path

HEAD = "{head_address}"

def project_root() -> Path:
    for key in ("CDSW_PROJECT_ROOT", "PROJECT_ROOT", "CDSW_WORKING_DIRECTORY"):
        val = os.environ.get(key)
        if val:
            p = Path(val)
            if p.is_dir():
                return p.resolve()
    return Path.cwd().resolve()

def ensure_ray(py: str, root: Path) -> None:
    import importlib.util
    if importlib.util.find_spec("ray") is not None:
        return
    req = root / "backend" / "requirements.txt"
    env = {{**os.environ, "PIP_USER": "0"}}
    if req.is_file():
        subprocess.check_call([py, "-m", "pip", "install", "-q", "-r", str(req)], env=env, cwd=str(root))
    else:
        subprocess.check_call([py, "-m", "pip", "install", "-q", "ray"], env=env)

py = sys.executable
root = project_root()
os.chdir(root)
ensure_ray(py, root)
print(f"[platform] CDSW worker connecting to Ray head (master) at {{HEAD}}", flush=True)
cmd = [py, "-m", "ray.scripts.scripts", "start", "--block", f"--address={{HEAD}}"]
rc = subprocess.call(cmd)
print(f"[platform] ray start --block exited with code {{rc}}", flush=True)
raise SystemExit(rc)
'''


def worker_launch_env(project_root: Path | None = None) -> dict[str, str]:
    root = (project_root or REPO_ROOT).resolve()
    env: dict[str, str] = {
        "PIP_USER": "0",
        "CDSW_PROJECT_ROOT": str(root),
        "PROJECT_ROOT": str(root),
    }
    for key in (
        "CDSW_WORKING_DIRECTORY",
        "HOME",
        "PATH",
        "PYTHONPATH",
        "CML_RESOURCE_PROFILE_ID",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


class CdswRayWorkerPool:
    """Provision GPU workers via cdsw/cml workers API and attach them to Ray."""

    def __init__(
        self,
        cpu: float = 4.0,
        memory_gb: float = 16.0,
        gpus_per_worker: int = 1,
        *,
        accelerator_label_id: str | None = None,
        resource_profile_id: str | None = None,
        resource_group_id: str | None = None,
    ) -> None:
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.gpus_per_worker = gpus_per_worker
        self._accelerator_label_id = accelerator_label_id
        self._resource_profile_id = resource_profile_id
        self._resource_group_id = resource_group_id
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
            "env": worker_launch_env(),
        }
        optional = {
            "accelerator_label_id": self._accelerator_label_id
            or os.environ.get("CML_ACCELERATOR_LABEL_ID"),
            "resource_profile_id": self._resource_profile_id
            or os.environ.get("CML_RESOURCE_PROFILE_ID"),
            "resource_group_id": self._resource_group_id
            or os.environ.get("CML_RESOURCE_GROUP_ID"),
        }
        for key, val in optional.items():
            if val is not None and str(val).strip():
                kwargs[key] = int(val)

        log_worker_launch_memory(self.memory_gb, self.gpus_per_worker)
        raw = self._client.launch_workers(**kwargs)
        worker_ids = _extract_worker_ids(raw)
        self._launched = LaunchedWorkers(worker_ids=worker_ids, raw=raw)
        logger.info("Launched %s CDSW workers for Ray head %s", count, head_address)
        return self._launched

    def list_workers_safe(self) -> list[dict[str, Any]]:
        if self._client is None:
            return []
        try:
            return self._client.list_workers()
        except Exception:  # noqa: BLE001
            logger.exception("list_workers failed")
            return []

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
