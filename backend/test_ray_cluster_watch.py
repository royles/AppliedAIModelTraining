from app.services.ray_cluster_watch import (
    parse_ray_status_active_nodes,
    parse_ray_status_resource_capacities,
)


def test_parse_ray_status_active_nodes():
    sample = """
-------- Active: --------
 1 node_6faacb29b11527f17a6203048f756847f90e5cc37e09398d774b4eaf

-------- Pending: --------
 (no pending nodes)

-------- Recent failures: --------
 (no failures)
"""
    assert parse_ray_status_active_nodes(sample) == 1


def test_parse_ray_status_active_nodes_multiline():
    sample = """Active:
 1 node_aaa
 2 node_bbb
Pending:
"""
    assert parse_ray_status_active_nodes(sample) == 2


def test_parse_ray_status_resource_capacities():
    sample = """
Resources
---------------------------------------------------------------
Total Usage:
 0.0/20.0 CPU
 0.0/1.0 GPU
 0B/11.39GiB memory
From request_resources:
 (none)
"""
    caps = parse_ray_status_resource_capacities(sample)
    assert caps["cpu"] == 20.0
    assert caps["gpu"] == 1.0
