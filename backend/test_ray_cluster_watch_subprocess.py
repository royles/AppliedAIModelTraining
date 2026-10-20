from app.services.ray_cluster_watch import count_alive_ray_nodes_subprocess


def test_count_alive_nodes_subprocess_no_cluster():
    assert count_alive_ray_nodes_subprocess("127.0.0.1:59999", timeout_sec=5.0) is None
