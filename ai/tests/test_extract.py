import json
import zipfile
from xml.sax.saxutils import escape

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.extract import DOCX_MIME, PDF_MIME, split_sentences
from app.main import app

client = TestClient(app)
HEADERS = {"X-Internal-Key": "test-internal-key"}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    # 테스트마다 임시 폴더를 UPLOAD_DIR, VECTOR_STORE_DIR로 쓴다
    upload = tmp_path / "uploads"
    store = tmp_path / "faiss"
    upload.mkdir()
    store.mkdir()
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload))
    monkeypatch.setattr(settings, "VECTOR_STORE_DIR", str(store))
    return upload, store


def make_docx(path, paragraphs):
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    body = "".join(f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


def post(body, headers=HEADERS):
    return client.post("/internal/extract", json=body, headers=headers)


# ---------- 문장 분리 ----------

def test_split_sentences_by_korean_endings():
    lines = ["결재는 팀장이 승인한다. 승인 기한은 3일이다. 문의는 총무팀에 하세요? 꼭 확인!"]
    assert split_sentences(lines) == [
        "결재는 팀장이 승인한다.",
        "승인 기한은 3일이다.",
        "문의는 총무팀에 하세요?",
        "꼭 확인!",
    ]


def test_line_without_punctuation_kept_as_one():
    assert split_sentences(["항목 금액 비고"]) == ["항목 금액 비고"]


# ---------- 정상 추출 ----------

def test_docx_extract_ok(dirs):
    upload, store = dirs
    make_docx(upload / "7" / "1" / "a.docx", [
        "출장비는 귀국 후 14일 이내 정산한다. 영수증을 첨부한다.",
        "3",  # 쪽번호처럼 숫자만 있는 줄은 지워져야 함
        "문의는 총무팀으로 하세요?",
    ])

    res = post({"versionId": 31, "filePath": "7/1/a.docx", "mimeType": DOCX_MIME})

    assert res.status_code == 200
    body = res.json()
    assert body["contentText"].split("\n") == [
        "출장비는 귀국 후 14일 이내 정산한다.",
        "영수증을 첨부한다.",
        "문의는 총무팀으로 하세요?",
    ]
    assert body["pageCount"] is None

    saved = json.loads((store / "_extract" / "31.json").read_text(encoding="utf-8"))
    assert saved[0]["pageNumber"] is None


# ---------- 경로 조작 ----------

@pytest.mark.parametrize("bad_path", [
    "../secret.docx",
    "7/../../secret.docx",
    "/etc/passwd",
    "C:/Windows/win.ini",
    "a\x00.docx",
])
def test_path_outside_upload_dir_rejected(dirs, bad_path):
    upload, _ = dirs
    make_docx(upload.parent / "secret.docx", ["밖에 있는 파일이다."])

    res = post({"versionId": 1, "filePath": bad_path, "mimeType": DOCX_MIME})

    assert res.status_code == 400
    assert res.json() == {"code": "INVALID_FILE"}


def test_missing_file_rejected(dirs):
    res = post({"versionId": 1, "filePath": "7/1/none.docx", "mimeType": DOCX_MIME})
    assert res.status_code == 400
    assert res.json() == {"code": "INVALID_FILE"}


# ---------- 형식·내용 ----------

def test_signature_mismatch_rejected(dirs):
    upload, _ = dirs
    make_docx(upload / "a.docx", ["본문이다."])
    # DOCX 파일인데 PDF라고 보냄
    res = post({"versionId": 1, "filePath": "a.docx", "mimeType": PDF_MIME})
    assert res.status_code == 400
    assert res.json() == {"code": "INVALID_FILE"}


def test_fake_pdf_rejected(dirs):
    upload, _ = dirs
    (upload / "fake.pdf").write_bytes(b"this is not a pdf")
    res = post({"versionId": 1, "filePath": "fake.pdf", "mimeType": PDF_MIME})
    assert res.status_code == 400


def test_empty_document_rejected(dirs):
    upload, _ = dirs
    make_docx(upload / "empty.docx", [])
    res = post({"versionId": 1, "filePath": "empty.docx", "mimeType": DOCX_MIME})
    assert res.status_code == 400
    assert res.json() == {"code": "INVALID_FILE"}


# ---------- 요청 검증·인증 ----------

def test_unsupported_mime_is_validation_error(dirs):
    res = post({"versionId": 1, "filePath": "a.txt", "mimeType": "text/plain"})
    assert res.status_code == 400
    assert res.json() == {"code": "VALIDATION_ERROR"}


def test_invalid_version_id_is_validation_error(dirs):
    res = post({"versionId": 0, "filePath": "a.docx", "mimeType": DOCX_MIME})
    assert res.status_code == 400
    assert res.json() == {"code": "VALIDATION_ERROR"}


def test_no_key_returns_401(dirs):
    res = post({"versionId": 1, "filePath": "a.docx", "mimeType": DOCX_MIME}, headers={})
    assert res.status_code == 401