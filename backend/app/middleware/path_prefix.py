"""Strip reverse-proxy path prefixes (CAI / X-Forwarded-Prefix) before routing."""

from __future__ import annotations

import os

from starlette.types import ASGIApp, Receive, Scope, Send


def _header_prefix(scope: Scope) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == b"x-forwarded-prefix":
            decoded = value.decode("latin-1").strip()
            if decoded:
                return decoded.rstrip("/")
    return None


def _effective_prefix(scope: Scope) -> str:
    env_prefix = (os.environ.get("PLATFORM_ROOT_PATH") or "").strip().rstrip("/")
    header = _header_prefix(scope)
    if header:
        return header
    return env_prefix


class PathPrefixMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        prefix = _effective_prefix(scope)
        if prefix and path.startswith(prefix):
            new_scope = dict(scope)
            stripped = path[len(prefix) :] or "/"
            new_scope["path"] = stripped
            new_scope["root_path"] = (scope.get("root_path") or "") + prefix
            await self.app(new_scope, receive, send)
            return

        await self.app(scope, receive, send)
