import logging

from fastapi import APIRouter, HTTPException, Request

from app.core.errors import InvalidFileError
from app.core.extract import extract, save_page_texts
from app.schemas.extract import ExtractRequest, ExtractResponse
from app.security.paths import resolve_upload_path

logger = logging.getLogger("ai.extract")
router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
def extract_text(req: ExtractRequest, request: Request) -> ExtractResponse:
    trace = request.headers.get("X-Trace-Id", "-")
    try:
        path = resolve_upload_path(req.file_path)
        result = extract(path, req.mime_type)
        save_page_texts(req.version_id, result.pages)
    except InvalidFileError as e:
        # 경로·본문은 로그에 남기지 않고, 고정된 사유 문구만 남긴다
        logger.warning("extract rejected trace=%s versionId=%s reason=%s", trace, req.version_id, e)
        raise HTTPException(status_code=400, detail="INVALID_FILE") from e

    logger.info(
        "extract ok trace=%s versionId=%s pages=%s sentences=%s",
        trace, req.version_id, result.page_count, result.content_text.count("\n") + 1,
    )
    return ExtractResponse(content_text=result.content_text, page_count=result.page_count)