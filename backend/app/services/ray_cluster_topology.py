"""Ray cluster topology for the task-detail monitoring UI."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from app.services.cai_networking import master_engine_ip


def _gcs_reachable(address: str | None, timeout: float = 1.5) -> bool:
    if not address:
        return False
    host, _, port_s = address.partition(":")
    port = int(port_s or "6379")
    if host in ("ray://",):
        return False
    host = host.replace("ray://", "").strip() or "127.0.0.1"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@contextmanager
def _ray_session(ray_address: str | None) -> Iterator[bool]:
    """Attach read-only to the job cluster for one snapshot."""
    if not ray_address:
        yield False
        return
    try:
        import ray
    except ImportError:
        yield False
        return

    already = ray.is_initialized()
    connected = already
    if not already:
        try:
            from app.services.cai_networking import coerce_session_driver_ray_address

            connect = coerce_session_driver_ray_address(ray_address) or ray_address
            if connect != ray_address:
                os.environ["RAY_ADDRESS"] = connect
            ray.init(address=connect, ignore_reinit_error=True, logging_level="error")
            connected = ray.is_initialized()
        except Exception:  # noqa: BLE001
            connected = False
    try:
        yield connected
    finally:
        if connected and not already:
            try:
                ray.shutdown()
            except Exception:  # noqa: BLE001
                pass


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pick_resources(node: dict[str, Any]) -> dict[str, float]:
    raw = node.get("Resources") or node.get("resources") or {}
    out: dict[str, float] = {}
    for key, val in raw.items():
        if str(key).startswith("node:"):
            continue
        out[str(key)] = _num(val)
    return out


def _format_bytes_gb(value: float) -> float:
    """Ray memory resources are often in bytes."""
    if value > 1_000_000:
        return round(value / (1024**3), 2)
    return round(value, 2)


def _node_metrics(resources: dict[str, float]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    order = ["CPU", "GPU", "memory", "object_store_memory"]
    seen: set[str] = set()
    for key in order:
        if key in resources:
            val = resources[key]
            label = key.replace("_", " ")
            unit = "GB" if "memory" in key.lower() else ""
            display = f"{_format_bytes_gb(val)} {unit}".strip() if unit else f"{val:g}"
            metrics.append(
                {
                    "key": key,
                    "label": label,
                    "value": val,
                    "display": display,
                    "unit": unit or "count",
                }
            )
            seen.add(key)
    for key in sorted(resources.keys()):
        if key in seen:
            continue
        val = resources[key]
        metrics.append(
            {
                "key": key,
                "label": key,
                "value": val,
                "display": f"{val:g}",
                "unit": "count",
            }
        )
    return metrics


def _is_head_node(node: dict[str, Any], master_ip: str | None) -> bool:
    addr = str(
        node.get("NodeManagerAddress")
        or node.get("node_ip_address")
        or node.get("NodeManagerHostname")
        or ""
    )
    if master_ip and master_ip in addr:
        return True
    resources = _pick_resources(node)
    # Session head often has no GPU but has object store.
    return resources.get("GPU", 0) == 0 and node.get("Alive", True)


def build_ray_cluster_topology(
    *,
    ray_address: str | None,
    job_status: str,
    expected_worker_nodes: int = 0,
    cdsw_worker_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Structured cluster map: master, optional train controller, worker nodes, links.
    """
    master_ip = master_engine_ip()
    gcs_ok = _gcs_reachable(ray_address)
    topology: dict[str, Any] = {
        "version": 1,
        "gcs_address": ray_address,
        "gcs_reachable": gcs_ok,
        "master_ip": master_ip,
        "job_status": job_status,
        "expected_cdsw_workers": expected_worker_nodes,
        "cdsw_worker_ids": cdsw_worker_ids or [],
        "components": [],
        "links": [],
        "summary": {
            "alive_nodes": 0,
            "dead_nodes": 0,
            "total_cpu": 0.0,
            "total_gpu": 0.0,
            "available_cpu": 0.0,
            "available_gpu": 0.0,
        },
    }

    def _add_component(comp: dict[str, Any]) -> str:
        topology["components"].append(comp)
        return comp["id"]

    def _link(from_id: str, to_id: str, *, healthy: bool, kind: str, detail: str = "") -> None:
        topology["links"].append(
            {
                "from": from_id,
                "to": to_id,
                "healthy": healthy,
                "kind": kind,
                "detail": detail,
            }
        )

    driver_id = _add_component(
        {
            "id": "platform-driver",
            "role": "driver",
            "title": "Training driver",
            "subtitle": "Session API / job runner",
            "healthy": gcs_ok,
            "host": ray_address or "local",
            "metrics": [],
        }
    )

    with _ray_session(ray_address) as connected:
        if not connected:
            head_id = _add_component(
                {
                    "id": "ray-head",
                    "role": "master",
                    "title": "Ray head (GCS)",
                    "subtitle": "Not connected from UI snapshot",
                    "healthy": gcs_ok,
                    "host": ray_address or master_ip or "unknown",
                    "metrics": [],
                }
            )
            _link(driver_id, head_id, healthy=gcs_ok, kind="gcs", detail="TCP 6379")
            if not gcs_ok:
                topology["note"] = (
                    "Cannot reach Ray GCS from the UI process. Cluster map updates when "
                    "the head is up and ray_address is set on the job."
                )
            return topology

        import ray

        try:
            total_res = {str(k): _num(v) for k, v in ray.cluster_resources().items()}
            avail_res = {str(k): _num(v) for k, v in ray.available_resources().items()}
        except Exception as exc:  # noqa: BLE001
            topology["cluster_error"] = str(exc)
            total_res = {}
            avail_res = {}

        nodes_raw = []
        try:
            nodes_raw = ray.nodes()
        except Exception as exc:  # noqa: BLE001
            topology["cluster_error"] = str(exc)

        alive = [n for n in nodes_raw if n.get("Alive")]
        dead = [n for n in nodes_raw if not n.get("Alive")]
        topology["summary"]["alive_nodes"] = len(alive)
        topology["summary"]["dead_nodes"] = len(dead)
        topology["summary"]["total_cpu"] = total_res.get("CPU", 0)
        topology["summary"]["total_gpu"] = total_res.get("GPU", 0)
        topology["summary"]["available_cpu"] = avail_res.get("CPU", 0)
        topology["summary"]["available_gpu"] = avail_res.get("GPU", 0)
        topology["summary"]["used_cpu"] = max(
            0, topology["summary"]["total_cpu"] - topology["summary"]["available_cpu"]
        )
        topology["summary"]["used_gpu"] = max(
            0, topology["summary"]["total_gpu"] - topology["summary"]["available_gpu"]
        )

        head_node = None
        worker_nodes: list[dict[str, Any]] = []
        for node in alive:
            if _is_head_node(node, master_ip) and head_node is None:
                head_node = node
            else:
                worker_nodes.append(node)
        if head_node is None and alive:
            head_node = alive[0]
            worker_nodes = alive[1:]

        head_addr = "unknown"
        if head_node:
            head_addr = str(
                head_node.get("NodeManagerAddress")
                or head_node.get("node_ip_address")
                or master_ip
                or "head"
            )
        head_resources = _pick_resources(head_node) if head_node else total_res
        head_id = _add_component(
            {
                "id": "ray-head",
                "role": "master",
                "title": "Ray head",
                "subtitle": "GCS · scheduler",
                "healthy": bool(head_node and head_node.get("Alive")),
                "host": head_addr,
                "node_id": (head_node or {}).get("NodeID"),
                "metrics": _node_metrics(head_resources),
            }
        )
        _link(driver_id, head_id, healthy=gcs_ok and bool(head_node), kind="gcs", detail="Driver → GCS")

        train_running = job_status in ("provisioning", "running", "queued")
        controller_healthy = train_running and topology["summary"]["total_gpu"] > 0
        controller_id = _add_component(
            {
                "id": "train-controller",
                "role": "controller",
                "title": "Ray Train",
                "subtitle": "Train controller",
                "healthy": controller_healthy if train_running else job_status == "succeeded",
                "host": head_addr,
                "metrics": [
                    {
                        "key": "status",
                        "label": "Job",
                        "value": 0,
                        "display": job_status,
                        "unit": "text",
                    },
                    {
                        "key": "gpu_pool",
                        "label": "GPU pool",
                        "value": topology["summary"]["total_gpu"],
                        "display": (
                            f"{topology['summary']['used_gpu']:g}/"
                            f"{topology['summary']['total_gpu']:g} in use"
                        ),
                        "unit": "count",
                    },
                ],
            }
        )
        _link(
            head_id,
            controller_id,
            healthy=controller_healthy or job_status == "succeeded",
            kind="train",
            detail="Head schedules Train workers",
        )

        for idx, node in enumerate(worker_nodes, start=1):
            node_id = str(node.get("NodeID") or f"worker-{idx}")
            short = node_id[:8]
            addr = str(
                node.get("NodeManagerAddress")
                or node.get("node_ip_address")
                or f"worker-{idx}"
            )
            resources = _pick_resources(node)
            wid = cdsw_worker_ids[idx - 1] if cdsw_worker_ids and idx - 1 < len(cdsw_worker_ids) else None
            comp_id = _add_component(
                {
                    "id": f"node-{short}",
                    "role": "worker",
                    "title": f"Worker {idx}",
                    "subtitle": f"CDSW {wid}" if wid else "Ray node",
                    "healthy": bool(node.get("Alive")),
                    "host": addr,
                    "node_id": node_id,
                    "cdsw_worker_id": wid,
                    "metrics": _node_metrics(resources),
                }
            )
            _link(
                head_id,
                comp_id,
                healthy=bool(node.get("Alive")),
                kind="worker",
                detail="Worker ↔ head",
            )
            _link(
                controller_id,
                comp_id,
                healthy=bool(node.get("Alive")) and resources.get("GPU", 0) > 0,
                kind="train_worker",
                detail="Train worker placement",
            )

        if expected_worker_nodes > len(worker_nodes):
            for missing in range(len(worker_nodes) + 1, expected_worker_nodes + 1):
                wid = (
                    cdsw_worker_ids[missing - 1]
                    if cdsw_worker_ids and missing - 1 < len(cdsw_worker_ids)
                    else None
                )
                ghost_id = _add_component(
                    {
                        "id": f"pending-worker-{missing}",
                        "role": "worker_pending",
                        "title": f"Worker {missing}",
                        "subtitle": "Expected · not in cluster",
                        "healthy": False,
                        "host": "pending",
                        "cdsw_worker_id": wid,
                        "metrics": [],
                    }
                )
                _link(head_id, ghost_id, healthy=False, kind="worker", detail="Awaiting ray start")
                _link(controller_id, ghost_id, healthy=False, kind="train_worker", detail="GPU not available")

        for node in dead:
            node_id = str(node.get("NodeID") or "dead")
            comp_id = _add_component(
                {
                    "id": f"dead-{node_id[:8]}",
                    "role": "worker_dead",
                    "title": "Node lost",
                    "subtitle": str(node.get("DeathReason") or "not alive"),
                    "healthy": False,
                    "host": str(node.get("NodeManagerAddress") or "?"),
                    "node_id": node_id,
                    "metrics": _node_metrics(_pick_resources(node)),
                }
            )
            _link(head_id, comp_id, healthy=False, kind="worker", detail="Node dead")

    return topology


def merge_topology_into_ray_status(
    ray_status: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any]:
    """Keep legacy flat fields for older UI + add topology."""
    merged = dict(ray_status)
    merged["topology"] = topology
    summary = topology.get("summary") or {}
    if summary:
        merged["nodes"] = summary.get("alive_nodes", merged.get("nodes"))
        merged["cluster_resources"] = {
            "CPU": summary.get("total_cpu", 0),
            "GPU": summary.get("total_gpu", 0),
        }
        merged["cluster_resources_available"] = {
            "CPU": summary.get("available_cpu", 0),
            "GPU": summary.get("available_gpu", 0),
        }
        merged["cluster_resources_used"] = {
            "CPU": summary.get("used_cpu", 0),
            "GPU": summary.get("used_gpu", 0),
        }
    merged["gcs_reachable"] = topology.get("gcs_reachable")
    return merged
