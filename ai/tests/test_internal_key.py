from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from app.main import app
from app.security.internal_key import verify_internal_key

# 아직 실제 /internal API가 없어서, 테스트용 경로를 하나 붙여서 검사한다
_test_router = APIRouter()


@_test_router.get("/internal/_test")
def _protected():
    return {"ok": True}


app.include_router(_test_router, dependencies=[Depends(verify_internal_key)])
client = TestClient(app)

VALID_KEY = "test-internal-key"  # conftest.py 에서 넣은 값


def test_no_key_returns_401():
    res = client.get("/internal/_test")
    assert res.status_code == 401
    assert res.json() == {"code": "UNAUTHORIZED"}


def test_wrong_key_returns_401():
    res = client.get("/internal/_test", headers={"X-Internal-Key": "wrong-key-wrong-key"})
    assert res.status_code == 401
    assert res.json() == {"code": "UNAUTHORIZED"}


def test_valid_key_passes():
    res = client.get("/internal/_test", headers={"X-Internal-Key": VALID_KEY})
    assert res.status_code == 200


def test_health_needs_no_key():
    assert client.get("/health").status_code == 200