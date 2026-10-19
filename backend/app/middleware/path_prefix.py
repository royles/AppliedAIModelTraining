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


def effective_path_prefix(scope: Scope) -> str:
    """Proxy path prefix from X-Forwarded-Prefix or PLATFORM_ROOT_PATH."""
    env_prefix = (os.environ.get("PLATFORM_ROOT_PATH") or "").strip().rstrip("/")
    header = _header_prefix(scope)
    if header:
        return header
    return env_prefix


def base_href_for_scope(scope: Scope) -> str:
    """HTML <base href> for relative ./assets/* URLs (CAI application subpaths)."""
    prefix = effective_path_prefix(scope).rstrip("/")
    return f"{prefix}/" if prefix else "/"


class PathPrefixMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        prefix = effective_path_prefix(scope)
        if prefix and path.startswith(prefix):
            new_scope = dict(scope)
            stripped = path[len(prefix) :] or "/"
            new_scope["path"] = stripped
            # Do not set scope["root_path"] here: we already rewrote path. Starlette
            # StaticFiles + Mount break when root_path is set but path is stripped
            # (assets 404 → blank React shell with only CSS background).
            await self.app(new_scope, receive, send)
            return

        await self.app(scope, receive, send)
