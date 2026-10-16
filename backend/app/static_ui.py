"""Serve production React build from frontend/dist (no Node/npm at runtime)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import REPO_ROOT


def frontend_dist_dir() -> Path:
    return REPO_ROOT / "frontend" / "dist"


def frontend_dist_ready() -> bool:
    dist = frontend_dist_dir()
    return dist.is_dir() and (dist / "index.html").is_file()


def mount_static_ui(app: FastAPI) -> bool:
    dist = frontend_dist_dir()
    if not frontend_dist_ready():
        return False
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
    return True


def read_ui_build_info() -> dict[str, str]:
    """Asset hashes from index.html + optional stamp for debugging stale CAI bundles."""
    dist = frontend_dist_dir()
    info: dict[str, str] = {"status": "missing"}
    if not frontend_dist_ready():
        return info
    index_path = dist / "index.html"
    try:
        html = index_path.read_text(encoding="utf-8")
    except OSError:
        return info
    import re

    js_match = re.search(r'/assets/(index-[^"\']+\.js)', html)
    css_match = re.search(r'/assets/(index-[^"\']+\.css)', html)
    stamp_path = dist / ".ui-build-stamp"
    stamp = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.is_file() else ""
    info = {
        "status": "ready",
        "js": js_match.group(1) if js_match else "",
        "css": css_match.group(1) if css_match else "",
        "stamp": stamp,
    }
    return info


def mount_gateway_spa(app: FastAPI) -> bool:
    """
    Serve the built React app without a catch-all / mount (keeps /api and /v1 routable).
    """
    dist = frontend_dist_dir()
    if not frontend_dist_ready():
        return False

    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="spa-assets",
        )

    index_path = dist / "index.html"

    def _spa_index_response() -> FileResponse:
        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/", include_in_schema=False)
    async def spa_index() -> FileResponse:
        return _spa_index_response()

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def spa_fallback(spa_path: str) -> FileResponse:
        blocked_prefixes = ("api/", "v1/", "assets/")
        blocked_exact = {"v1", "assets"}
        if spa_path in blocked_exact or any(spa_path.startswith(p) for p in blocked_prefixes):
            raise HTTPException(status_code=404, detail="Not found")
        return _spa_index_response()

    return True
