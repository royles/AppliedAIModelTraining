"""Console logging defaults: errors only unless PLATFORM_LOG_LEVEL is set."""

from __future__ import annotations

import logging
import os

_QUIET_LIBRARIES = (
    "urllib3",
    "httpx",
    "httpcore",
    "datasets",
    "filelock",
    "botocore",
    "s3transfer",
    "fsspec",
    "transformers",
    "ray",
    "ray.train",
    "uvicorn.access",
)


def configure_platform_logging() -> None:
    level_name = os.environ.get("PLATFORM_LOG_LEVEL", "ERROR").strip().upper()
    level = getattr(logging, level_name, logging.ERROR)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    root = logging.getLogger()
    root.setLevel(level)
    if level > logging.DEBUG:
        for name in _QUIET_LIBRARIES:
            logging.getLogger(name).setLevel(logging.ERROR)
