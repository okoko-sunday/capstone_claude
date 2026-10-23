import importlib

import pytest
from fastapi.testclient import TestClient

from tests.conftest import jpeg_bytes


def make_client(monkeypatch, model_dir):
    monkeypatch.setenv("CATDOG_MODEL_DIR", str(model_dir))
    import catdog.server.app as app_module

    app_module = importlib.reload(app_module)
    return TestClient(app_module.app)


@pytest.fixture
def client(monkeypatch, tiny_model_dir):
    with make_client(monkeypatch, tiny_model_dir) as c:
        yield c


def test_health_and_model_card(client):
    assert client.get("/api/health").json()["status"] == "ok"
    card = client.get("/api/model").json()
    assert card["labels"] == ["cat", "dog", "neither"] and card["image_size"] == 32


def test_predict_mixed_upload(client):
    files = [
        ("files", ("dog/one.jpg", jpeg_bytes(800, 1200), "image/jpeg")),
        ("files", ("broken.jpg", b"nope", "image/jpeg")),
        ("files", (".DS_Store", b"junk", "application/octet-stream")),
    ]
    body = client.post("/api/predict", files=files).json()
    assert [p["filename"] for p in body["predictions"]] == ["dog/one.jpg", "broken.jpg"]
    assert body["predictions"][1]["label"] == "error"
    assert sum(body["counts"].values()) == 2


def test_predict_rejects_no_images(client):
    r = client.post("/api/predict", files=[("files", ("a.txt", b"x", "text/plain"))])
    assert r.status_code == 400


def test_evaluate(client):
    items = [{"true": "dog", "predicted": "dog"}, {"true": "cat", "predicted": "neither"}, {"true": "cat", "predicted": "error"}]
    r = client.post("/api/evaluate", json={"items": items}).json()
    assert r["total"] == 2 and r["correct"] == 1 and r["errors_excluded"] == 1


def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Cat, Dog" in r.text
    assert client.get("/assets/app.js").status_code == 200


def test_no_model_degrades_gracefully(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path) as c:
        assert c.get("/api/health").json()["status"] == "no_model"
        r = c.post("/api/predict", files=[("files", ("a.jpg", jpeg_bytes(), "image/jpeg"))])
        assert r.status_code == 503
