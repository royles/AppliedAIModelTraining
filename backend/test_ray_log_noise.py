from app.services.ray_log_noise import is_ray_train_noise_line


def test_placement_group_cleaner_is_noise():
    assert is_ray_train_noise_line(
        "(PlacementGroupCleaner pid=1) Failed to query Ray Train Controller actor state. "
        "State API may be temporarily unavailable. Continuing to monitor."
    )


def test_sync_actor_report_wait_is_noise():
    assert is_ray_train_noise_line(
        "`ray.train.report` has not been called by all 4 workers in the group."
    )


def test_training_step_is_not_noise():
    assert not is_ray_train_noise_line("step 12/100 loss=1.23 lr=3e-5")
