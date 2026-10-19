"""UI build-info API response shape."""

from fastapi.testclient import TestClient

from app.main import app


def test_ui_build_info_accepts_features_object() -> None:
    client = TestClient(app)
    response = client.get("/api/ui/build-info")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    if body["status"] == "ready":
        assert isinstance(body.get("features"), dict)
        assert "model_registry" in body["features"]
