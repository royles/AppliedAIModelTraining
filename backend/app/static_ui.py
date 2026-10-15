"""Serve production React build from frontend/dist (no Node/npm at runtime)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
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
