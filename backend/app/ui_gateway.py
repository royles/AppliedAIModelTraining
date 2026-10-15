"""UX entrypoint for CAI: React static files + same FastAPI routes as app.main (no HTTP proxy)."""

from __future__ import annotations

from app.main import create_app
from app.static_ui import frontend_dist_ready, mount_gateway_spa


def create_ui_app():
    if not frontend_dist_ready():
        raise RuntimeError("frontend/dist is missing; run scripts/install.py")

    app = create_app()
    mount_gateway_spa(app)
    return app


app = create_ui_app()
