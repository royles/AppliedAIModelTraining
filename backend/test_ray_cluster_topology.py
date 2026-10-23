"""Topology builder must not ray.init in the API process."""

from __future__ import annotations

from unittest.mock import patch

from app.services.ray_cluster_topology import build_ray_cluster_topology


def test_topology_skips_ray_init_when_gcs_tcp_down(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("ray.init must not run when GCS TCP is down")

    monkeypatch.setattr(
        "app.services.ray_cluster_topology._gcs_reachable",
        lambda *_a, **_k: False,
    )
    with patch("ray.init", side_effect=_boom):
        topo = build_ray_cluster_topology(
            ray_address="100.100.191.24:6379",
            job_status="running",
            expected_worker_nodes=2,
        )
    assert topo["gcs_reachable"] is False
    assert topo["summary"]["alive_nodes"] == 0


def test_topology_uses_ray_status_when_not_active_job(monkeypatch):
    monkeypatch.setattr(
        "app.services.ray_cluster_topology._gcs_reachable",
        lambda *_a, **_k: True,
    )
    with patch(
        "app.services.ray_gcs_auth.verified_gcs_auth_token",
        return_value="tok",
    ):
        with patch(
            "app.services.ray_cluster_watch.ray_status_cluster_snapshot",
            return_value={"alive": 2.0, "gpu": 1.0, "cpu": 8.0},
        ):
            with patch(
                "app.services.ray_cluster_watch.ray_cluster_topology_subprocess",
                side_effect=AssertionError("subprocess init only for active jobs"),
            ):
                with patch("ray.init", side_effect=AssertionError("no in-process init")):
                    topo = build_ray_cluster_topology(
                        ray_address="100.100.191.24:6379",
                        job_status="succeeded",
                    )
    assert topo["summary"]["alive_nodes"] == 2
    assert topo["summary"]["total_gpu"] == 1.0
