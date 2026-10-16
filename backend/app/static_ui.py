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
        return FileResponse(index_path)

    @app.get("/", include_in_schema=False)
    async def spa_index() -> FileResponse:
        return _spa_index_response()

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def spa_fallback(spa_path: str) -> FileResponse:
        blocked_prefixes = ("api/", "v1/", "assets/")
        blocked_exact = {"api", "v1", "assets", "docs", "redoc", "openapi.json"}
        if spa_path in blocked_exact or any(spa_path.startswith(p) for p in blocked_prefixes):
            raise HTTPException(status_code=404, detail="Not found")
        return _spa_index_response()

    return True
