from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_unknown_path_returns_code_only():
    res = client.get("/no-such-path")
    assert res.status_code == 404
    assert res.json() == {"code": "NOT_FOUND"}


def test_docs_disabled_by_default():
    assert client.get("/docs").status_code == 404