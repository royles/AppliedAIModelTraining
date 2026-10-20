from app.services.ray_cluster_watch import parse_ray_status_active_nodes


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
