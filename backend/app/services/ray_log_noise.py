"""Filter known-benign Ray Train log noise on driver and workers."""

from __future__ import annotations

import logging
import re

_PLACEMENT_GROUP_CLEANER = re.compile(
    r"Failed to query Ray Train Controller actor state",
    re.I,
)
_SYNC_REPORT_WAIT = re.compile(
    r"`ray\.train\.report` has not been called by all \d+ workers",
    re.I,
)
_SYNC_BARRIER_HINT = re.compile(
    r"workers are not hanging on other operations.*synchronization barrier",
    re.I,
)
_SYNC_ENV_HINT = re.compile(
    r"RAY_TRAIN_COLLECTIVE_WARN_INTERVAL_S",
    re.I,
)

_RAY_TRAIN_LOGGER_PREFIXES = (
    "ray.train.v2._internal.execution.controller.placement_group_cleaner",
    "ray.train.v2._internal.execution.checkpoint.sync_actor",
    "ray.train.v2._internal.util",
)


class _RayTrainNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if _PLACEMENT_GROUP_CLEANER.search(msg):
            return False
        if _SYNC_REPORT_WAIT.search(msg):
            return False
        if _SYNC_BARRIER_HINT.search(msg):
            return False
        if _SYNC_ENV_HINT.search(msg) and "following ranks to join" in msg:
            return False
        if "waiting for the following ranks to join" in msg.lower():
            return False
        return True


_NOISE_FILTER = _RayTrainNoiseFilter()
_applied = False


def apply_ray_train_log_filters() -> None:
    """Drop PlacementGroupCleaner State API warnings and collective wait spam."""
    global _applied
    if _applied:
        return
    _applied = True
    for name in _RAY_TRAIN_LOGGER_PREFIXES:
        logging.getLogger(name).addFilter(_NOISE_FILTER)
    logging.getLogger("ray.train").addFilter(_NOISE_FILTER)


def is_ray_train_noise_line(message: str) -> bool:
    """For stored log streams (Ray may print outside Python logging)."""
    if _PLACEMENT_GROUP_CLEANER.search(message):
        return True
    if _SYNC_REPORT_WAIT.search(message):
        return True
    if "SynchronizationActor" in message and _SYNC_ENV_HINT.search(message):
        return True
    if "SynchronizationActor" in message and "following ranks to join" in message:
        return True
    return False
