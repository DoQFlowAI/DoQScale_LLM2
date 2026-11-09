import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.core import models
from app.core.config import settings
from app.core.vector_store import vector_store
from app.main import app

client = TestClient(app)
HEADERS = {"X-Internal-Key": "test-internal-key"}


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    d = tmp_path / "faiss"
    d.mkdir()
    monkeypatch.setattr(settings, "VECTOR_STORE_DIR", str(d))
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setattr(settings, "CHUNK_SIZE", 50)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 0)
    # 실제 OpenAI 대신 가짜 임베딩 (같은 글자면 같은 벡터)
    monkeypatch.setattr(models, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    vector_store.reset()
    yield d
    vector_store.reset()


def write_extract(store_dir, version_id, pages):
    out = store_dir / "_extract"
    out.mkdir(exist_ok=True)
    (out / f"{version_id}.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")


SAMPLE_PAGES = [
    {"pageNumber": 1, "text": "출장비는 귀국 후 14일 이내 정산한다.\n영수증을 반드시 첨부한다."},
    {"pageNumber": 2, "text": "문의는 총무팀으로 한다."},
]


def index(store_dir, document_id, version_id, version_no, pages=SAMPLE_PAGES):
    write_extract(store_dir, version_id, pages)
    return client.post(
        "/internal/index",
        json={"documentId": document_id, "versionId": version_id, "versionNo": version_no},
        headers=HEADERS,
    )


def delete(body):
    # httpx의 delete()는 json 본문을 못 받아서 request()로 보낸다
    return client.request("DELETE", "/internal/index", json=body, headers=HEADERS)


def version_ids():
    return {m["versionId"] for m in vector_store.all_metadata()}


# ---------- 색인 ----------

def test_index_ok(store_dir):
    res = index(store_dir, 7, 33, 3)

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "READY"
    assert body["chunkCount"] >= 2

    metas = vector_store.all_metadata()
    assert len(metas) == body["chunkCount"]
    assert {m["pageNumber"] for m in metas} == {1, 2}  # 페이지 번호 유지
    assert all(m["documentId"] == 7 and m["versionId"] == 33 and m["versionNo"] == 3 for m in metas)
    assert sorted(m["chunkIndex"] for m in metas) == list(range(len(metas)))

    assert not (store_dir / "_extract" / "33.json").exists()  # 임시 파일 삭제됨
    assert (store_dir / "test-embedding" / "index.faiss").exists()  # 모델명 폴더에 저장됨


def test_metadata_has_no_permission_info(store_dir):
    index(store_dir, 7, 33, 3)
    allowed_keys = {"versionId", "documentId", "versionNo", "chunkIndex", "pageNumber"}
    assert all(set(m) == allowed_keys for m in vector_store.all_metadata())


def test_missing_extract_file_rejected(store_dir):
    res = client.post(
        "/internal/index",
        json={"documentId": 7, "versionId": 99, "versionNo": 1},
        headers=HEADERS,
    )
    assert res.status_code == 400
    assert res.json() == {"code": "INVALID_FILE"}


def test_reindex_same_version_does_not_duplicate(store_dir):
    first = index(store_dir, 7, 33, 3).json()["chunkCount"]
    second = index(store_dir, 7, 33, 3).json()["chunkCount"]
    assert first == second
    assert len(vector_store.all_metadata()) == second


def test_index_persisted_after_restart(store_dir):
    index(store_dir, 7, 33, 3)
    count = len(vector_store.all_metadata())

    vector_store.reset()  # 재시작 흉내
    vector_store.load()

    assert len(vector_store.all_metadata()) == count


def test_embedding_failure_returns_502_and_keeps_extract(store_dir, monkeypatch):
    class BrokenEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("down")

    monkeypatch.setattr(models, "get_embeddings", lambda: BrokenEmbeddings())
    res = index(store_dir, 7, 33, 3)

    assert res.status_code == 502
    assert res.json() == {"code": "AI_UNAVAILABLE"}
    assert (store_dir / "_extract" / "33.json").exists()  # 재시도할 수 있게 남아 있음


# ---------- 삭제 ----------

def test_delete_keeps_given_version(store_dir):
    index(store_dir, 7, 33, 3)
    index(store_dir, 7, 34, 4)
    index(store_dir, 8, 50, 1)
    count_33 = sum(1 for m in vector_store.all_metadata() if m["versionId"] == 33)

    res = delete({"documentId": 7, "keepVersionId": 34})

    assert res.status_code == 200
    assert res.json() == {"deleted": count_33}
    assert version_ids() == {34, 50}  # 다른 문서(8)는 그대로


def test_delete_all_of_document(store_dir):
    index(store_dir, 7, 33, 3)
    index(store_dir, 8, 50, 1)

    res = delete({"documentId": 7, "keepVersionId": None})

    assert res.status_code == 200
    assert version_ids() == {50}


def test_delete_without_keep_field_is_rejected(store_dir):
    index(store_dir, 7, 33, 3)
    res = delete({"documentId": 7})
    assert res.status_code == 400
    assert res.json() == {"code": "VALIDATION_ERROR"}
    assert version_ids() == {33}  # 아무것도 안 지워짐


def test_delete_on_empty_store(store_dir):
    res = delete({"documentId": 7, "keepVersionId": None})
    assert res.status_code == 200
    assert res.json() == {"deleted": 0}


# ---------- 헬스체크 ----------

def test_health_503_when_index_load_failed(store_dir):
    vector_store.failed = True
    res = client.get("/health")
    assert res.status_code == 503