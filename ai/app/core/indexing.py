"""추출 결과를 조각(chunk)으로 나누고, 추출 임시 파일을 관리한다 (명세서 4.1, 4.2, 5.1 ⑤)."""

import json
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.errors import InvalidFileError

# 추출 임시 파일 보관 시간. 업로드 → 추출 → 요약 → 색인은 보통 몇 분 안에 끝난다
EXTRACT_TTL_SECONDS = 60 * 60


def _extract_dir() -> Path:
    return Path(settings.VECTOR_STORE_DIR) / "_extract"


def _extract_path(version_id: int) -> Path:
    return _extract_dir() / f"{version_id}.json"


def load_page_texts(version_id: int) -> list[dict]:
    path = _extract_path(version_id)
    if not path.is_file():
        # 컨테이너 재시작·시간 초과 정리 등으로 없어진 경우 → Backend가 /internal/extract부터 다시 호출
        raise InvalidFileError("extract file missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InvalidFileError("extract file broken") from e


def build_chunks(
    pages: list[dict], document_id: int, version_id: int, version_no: int
) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    docs: list[Document] = []
    ids: list[str] = []

    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        # 페이지마다 따로 나눠서, 한 조각이 두 페이지에 걸치지 않게 한다
        for chunk in splitter.split_text(text):
            chunk_index = len(docs)
            docs.append(Document(
                page_content=chunk,
                # 권한·직급 정보는 넣지 않는다 (명세서 6장). ID만.
                metadata={
                    "versionId": version_id,
                    "documentId": document_id,
                    "versionNo": version_no,
                    "chunkIndex": chunk_index,
                    "pageNumber": page.get("pageNumber"),
                },
            ))
            ids.append(f"{version_id}:{chunk_index}")

    if not docs:
        raise InvalidFileError("no chunks")
    return docs, ids


def delete_page_texts(version_id: int) -> None:
    _extract_path(version_id).unlink(missing_ok=True)


def cleanup_stale_extracts(max_age_seconds: int = EXTRACT_TTL_SECONDS) -> int:
    """오래된 추출 임시 파일을 지운다. 지운 개수를 돌려준다.

    색인은 마지막 버전만 하므로, 중간 버전의 임시 파일은 index 성공으로 지워지지 않는다.
    문서 본문이 디스크에 계속 남지 않도록 색인 성공 후와 서버 기동 시 정리한다.
    """
    folder = _extract_dir()
    if not folder.is_dir():
        return 0

    cutoff = time.time() - max_age_seconds
    removed = 0
    for path in folder.iterdir():
        if not path.is_file() or path.suffix not in (".json", ".tmp"):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue  # 다른 요청이 먼저 지운 경우 등
    return removed