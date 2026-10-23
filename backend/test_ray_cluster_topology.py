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


def test_topology_skips_live_probes_for_succeeded_job(monkeypatch):
    monkeypatch.setattr(
        "app.services.ray_cluster_topology._gcs_reachable",
        lambda *_a, **_k: True,
    )
    with patch(
        "app.services.ray_gcs_auth.verified_gcs_auth_token",
        side_effect=AssertionError("no GCS probes for finished jobs"),
    ):
        topo = build_ray_cluster_topology(
            ray_address="100.100.191.24:6379",
            job_status="succeeded",
        )
    assert topo["summary"]["alive_nodes"] == 0
    assert "finished tasks" in (topo.get("note") or "")


def test_topology_skips_gcs_probes_for_stopped_job(monkeypatch):
    calls: list[str] = []

    def _track(name: str):
        def _inner(*_a, **_k):
            calls.append(name)
            return "tok"

        return _inner

    monkeypatch.setattr(
        "app.services.ray_cluster_topology._gcs_reachable",
        lambda *_a, **_k: True,
    )
    with patch(
        "app.services.ray_gcs_auth.verified_gcs_auth_token",
        side_effect=_track("verified_gcs_auth_token"),
    ):
        topo = build_ray_cluster_topology(
            ray_address="100.100.191.31:6379",
            job_status="stopped",
        )
    assert "verified_gcs_auth_token" not in calls
    assert "finished tasks" in (topo.get("note") or "")
