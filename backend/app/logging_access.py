"""Log each HTTP route once in the terminal; UI polls every 3s otherwise flood the console."""

from __future__ import annotations

import logging
import os
import re

_REQUEST = re.compile(r'"(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD) ([^ ]+) HTTP')
_JOB_ID = re.compile(r"/api/jobs/\d+")


def _route_key(access_message: str) -> str:
    match = _REQUEST.search(access_message)
    if not match:
        return access_message
    method, path = match.group(1), match.group(2)
    path = _JOB_ID.sub("/api/jobs/{id}", path)
    return f"{method} {path}"


class LogEachRouteOnceFilter(logging.Filter):
    """Emit the first access log per normalized route; suppress 3s polling repeats."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if os.environ.get("PLATFORM_ACCESS_LOG", "").lower() not in ("1", "true", "yes"):
            return False
        key = _route_key(record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True
