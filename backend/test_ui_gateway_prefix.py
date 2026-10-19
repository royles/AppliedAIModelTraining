"""Regression: CAI path prefix must not break /assets (blank UI)."""

from starlette.testclient import TestClient

from app.ui_gateway import create_ui_app


def test_assets_load_behind_forwarded_prefix() -> None:
    client = TestClient(create_ui_app())
    headers = {"X-Forwarded-Prefix": "/cdsw/app123"}

    index = client.get("/cdsw/app123/", headers=headers)
    assert index.status_code == 200
    assert '<base href="/cdsw/app123/">' in index.text

    js = client.get("/cdsw/app123/assets/index-r_26uw0I.js", headers=headers)
    assert js.status_code == 200
    assert "javascript" in (js.headers.get("content-type") or "")

    deep = client.get("/cdsw/app123/registry", headers=headers)
    assert deep.status_code == 200
    assert '<base href="/cdsw/app123/">' in deep.text
