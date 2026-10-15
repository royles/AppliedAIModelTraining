"""UX entrypoint for CAI: static React on CDSW_APP_PORT, proxy API from PLATFORM_API_PORT."""

from __future__ import annotations

import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from app.static_ui import frontend_dist_dir, frontend_dist_ready

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def create_ui_app() -> FastAPI:
    if not frontend_dist_ready():
        raise RuntimeError("frontend/dist is missing; run scripts/install.py")

    app = FastAPI(title="Applied AI Model Training UI", docs_url=None, redoc_url=None)

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy_api(path: str, request: Request) -> Response:
        return await _proxy(request, f"/api/{path}")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy_v1(path: str, request: Request) -> Response:
        return await _proxy(request, f"/v1/{path}")

    dist = frontend_dist_dir()
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
    return app


async def _proxy(request: Request, path: str) -> Response:
    url = f"{API_UPSTREAM}{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        upstream = await client.request(
            request.method,
            url,
            content=body,
            headers=headers,
        )
    resp_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in HOP_BY_HOP
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


app = create_ui_app()
