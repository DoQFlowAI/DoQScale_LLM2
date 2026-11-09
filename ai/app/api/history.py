import logging

from fastapi import APIRouter, HTTPException, Request

from app.graphs.history_graph import run_history
from app.schemas.history import HistoryRequest, HistoryResponse, HistorySegmentResult

logger = logging.getLogger("ai.history")
router = APIRouter()


@router.post("/history", response_model=HistoryResponse)
def summarize_history(req: HistoryRequest, request: Request) -> HistoryResponse:
    trace = request.headers.get("X-Trace-Id", "-")
    segments = [
        {"from": s.from_, "to": s.to, "blocks": [b.model_dump() for b in s.blocks]}
        for s in req.segments
    ]

    try:
        results = run_history(segments)
    except Exception as e:
        # 구간별 실패는 그래프 안에서 FAILED로 처리됨. 여기 오는 건 그래프 자체 오류
        logger.warning("history graph failed trace=%s documentId=%s error=%s",
                       trace, req.document_id, type(e).__name__)
        raise HTTPException(status_code=502, detail="AI_UNAVAILABLE") from e

    failed = sum(1 for r in results if r["summaryStatus"] == "FAILED")
    logger.info("history done trace=%s documentId=%s segments=%s failed=%s",
                trace, req.document_id, len(results), failed)
    return HistoryResponse(segments=[HistorySegmentResult.model_validate(r) for r in results])