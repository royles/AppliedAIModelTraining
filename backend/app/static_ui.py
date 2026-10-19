"""Serve production React build from frontend/dist (no Node/npm at runtime)."""

from __future__ import annotations

import html
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import REPO_ROOT
from app.middleware.path_prefix import base_href_for_scope
from app.platform_version import read_platform_version


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

    js_match = re.search(r'(?:\./|/)assets/(index-[^"\']+\.js)', html)
    css_match = re.search(r'(?:\./|/)assets/(index-[^"\']+\.css)', html)
    stamp_path = dist / ".ui-build-stamp"
    stamp = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.is_file() else ""
    js_name = js_match.group(1) if js_match else ""
    features: dict[str, bool] = {}
    if js_name:
        js_path = dist / "assets" / js_name
        try:
            bundle = js_path.read_text(encoding="utf-8", errors="ignore")
            features["model_registry"] = "Model registry" in bundle and "/api/registry" in bundle
        except OSError:
            features["model_registry"] = False
    info = {
        "status": "ready",
        "platform_version": read_platform_version(),
        "js": js_name,
        "css": css_match.group(1) if css_match else "",
        "stamp": stamp,
        "features": features,
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
    index_template = index_path.read_text(encoding="utf-8")

    def _spa_index_html(request: Request) -> str:
        base_href = html.escape(base_href_for_scope(request.scope), quote=True)
        html_out = index_template
        if re.search(r"<base\s", html_out, re.IGNORECASE):
            html_out = re.sub(
                r"<base\s[^>]*href=[^>]*>",
                f'<base href="{base_href}">',
                html_out,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            html_out = html_out.replace("<head>", f'<head>\n    <base href="{base_href}">', 1)
        return html_out

    def _spa_index_response(request: Request) -> HTMLResponse:
        return HTMLResponse(
            _spa_index_html(request),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/", include_in_schema=False)
    async def spa_index(request: Request) -> HTMLResponse:
        return _spa_index_response(request)

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def spa_fallback(spa_path: str, request: Request) -> HTMLResponse:
        blocked_prefixes = ("api/", "v1/", "assets/")
        blocked_exact = {"v1", "assets"}
        if spa_path in blocked_exact or any(spa_path.startswith(p) for p in blocked_prefixes):
            raise HTTPException(status_code=404, detail="Not found")
        if "/assets/" in spa_path or spa_path.endswith((".js", ".css", ".map", ".ico", ".svg")):
            raise HTTPException(status_code=404, detail="Not found")
        return _spa_index_response(request)

    return True
