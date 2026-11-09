import logging

from fastapi import APIRouter, HTTPException, Request

from app.core.errors import AIUnavailableError, InvalidFileError
from app.core.indexing import build_chunks, delete_page_texts, load_page_texts
from app.core.vector_store import vector_store
from app.schemas.index import DeleteIndexRequest, DeleteIndexResponse, IndexRequest, IndexResponse

logger = logging.getLogger("ai.index")
router = APIRouter()


@router.post("/index", response_model=IndexResponse)
def index_version(req: IndexRequest, request: Request) -> IndexResponse:
    trace = request.headers.get("X-Trace-Id", "-")
    try:
        pages = load_page_texts(req.version_id)
        docs, ids = build_chunks(pages, req.document_id, req.version_id, req.version_no)
        vector_store.replace_version(docs, ids, req.version_id)
    except InvalidFileError as e:
        logger.warning("index rejected trace=%s versionId=%s reason=%s", trace, req.version_id, e)
        raise HTTPException(status_code=400, detail="INVALID_FILE") from e
    except AIUnavailableError as e:
        logger.warning("index failed trace=%s versionId=%s reason=%s", trace, req.version_id, e)
        raise HTTPException(status_code=502, detail="AI_UNAVAILABLE") from e

    # 색인이 끝난 뒤에만 임시 파일 삭제 (실패하면 남겨 둬서 재시도 가능)
    delete_page_texts(req.version_id)
    logger.info("index ok trace=%s documentId=%s versionId=%s chunks=%s",
                trace, req.document_id, req.version_id, len(docs))
    return IndexResponse(status="READY", chunk_count=len(docs))


@router.delete("/index", response_model=DeleteIndexResponse)
def delete_index(req: DeleteIndexRequest, request: Request) -> DeleteIndexResponse:
    trace = request.headers.get("X-Trace-Id", "-")
    document_id = req.document_id
    keep = req.keep_version_id

    def should_delete(md: dict) -> bool:
        if md.get("documentId") != document_id:
            return False
        return keep is None or md.get("versionId") != keep

    try:
        deleted = vector_store.delete_where(should_delete)
    except AIUnavailableError as e:
        logger.warning("index delete failed trace=%s documentId=%s reason=%s", trace, document_id, e)
        raise HTTPException(status_code=502, detail="AI_UNAVAILABLE") from e

    logger.info("index delete trace=%s documentId=%s keepVersionId=%s deleted=%s",
                trace, document_id, keep, deleted)
    return DeleteIndexResponse(deleted=deleted)