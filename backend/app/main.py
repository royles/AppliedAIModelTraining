import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.openai_routes import router as openai_router
from app.api.registry_routes import router as registry_router
from app.api.routes import router
from app.config import REPO_ROOT, effective_cors_origins, get_settings
from app.database import SessionLocal, init_db
from app.services.job_lifecycle import reconcile_stale_jobs
from app.services.model_registry import backfill_registry_entries
from app.services.ray_gcs_auth import sync_ray_gcs_auth_token
from app.services.cai_networking import (
    apply_master_session_ray_driver_env,
    reconcile_live_ray_head_env,
)
from app.services.platform_settings import sync_platform_ray_head_from_session
from app.middleware.path_prefix import PathPrefixMiddleware
from app.logging_config import configure_platform_logging
from app.static_ui import mount_static_ui

configure_platform_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    init_db()
    live_head = reconcile_live_ray_head_env()
    if live_head:
        from app.services.cai_networking import gcs_tcp_reachable, parse_gcs_host_port
        from app.services.ray_gcs_auth import invalidate_stored_gcs_auth_tokens

        host, port = parse_gcs_host_port(live_head)
        if gcs_tcp_reachable(host, port, timeout_sec=1.5):
            sync_ray_gcs_auth_token(live_head, required=False)
        else:
            invalidate_stored_gcs_auth_tokens()
            os.environ.pop("RAY_AUTH_TOKEN", None)
    driver_ray = apply_master_session_ray_driver_env()
    if live_head:
        logging.getLogger(__name__).info("CAI live Ray head (workers): %s", live_head)
    if driver_ray:
        logging.getLogger(__name__).info("CAI master session Ray driver address: %s", driver_ray)
    db = SessionLocal()
    try:
        sync_platform_ray_head_from_session(db)
        backfill_registry_entries(db)
        reconcile_stale_jobs(
            db,
            reason=(
                "Framework restarted: training was interrupted. "
                "Use Restart on this task to run again (optionally resume from the last epoch)."
            ),
        )
    finally:
        db.close()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Applied AI Model Training",
        description="Ray + Cloudera CDSW workers platform for transformer training",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    origins = effective_cors_origins(settings.cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/docs", include_in_schema=False)
    async def legacy_docs_redirect() -> RedirectResponse:
        return RedirectResponse(url="/api", status_code=307)

    app.include_router(router, prefix="/api")
    app.include_router(registry_router, prefix="/api")
    app.include_router(openai_router, prefix="/v1")
    # UX is served on CDSW_APP_PORT via app.ui_gateway; optional combined mode for local dev.
    if os.environ.get("PLATFORM_EMBED_UI", "").lower() in ("1", "true", "yes"):
        mount_static_ui(app)
    # Outermost: strip CAI / proxy path prefix before route matching.
    app.add_middleware(PathPrefixMiddleware)
    return app


app = create_app()
