import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.chains.rag_chain import retrieve, stream_answer
from app.core.errors import AIUnavailableError
from app.schemas.chat import ChatRequest

logger = logging.getLogger("ai.chat")
router = APIRouter()


def _sse(data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    trace = request.headers.get("X-Trace-Id", "-")
    started = time.monotonic()

    # 검색은 스트림 시작 전에 끝낸다 → 실패하면 일반 502 응답으로 돌려줄 수 있다
    try:
        chunks, top_score = await run_in_threadpool(retrieve, req.question, req.scope, req.top_k)
    except AIUnavailableError as e:
        logger.warning("chat search failed trace=%s reason=%s", trace, e)
        raise HTTPException(status_code=502, detail="AI_UNAVAILABLE") from e

    async def events():
        outcome = "answered"
        try:
            if not chunks:
                # 기준을 넘는 조각 없음 → LLM 호출 안 함
                outcome = "no_chunks"
                yield _sse({"noAnswer": True})
            else:
                async for event in stream_answer(req.question, chunks):
                    if "noAnswer" in event:
                        outcome = "no_answer"
                    yield _sse(event)
        except Exception as e:
            # 스트림이 시작된 뒤라 상태 코드를 바꿀 수 없음 → error 이벤트로 알림
            outcome = "error"
            logger.warning("chat stream failed trace=%s error=%s", trace, type(e).__name__)
            yield _sse({"error": "AI_UNAVAILABLE"})

        yield _sse("[DONE]")
        # 질문·답변·조각 원문은 남기지 않는다 (명세서 9장)
        logger.info(
            "chat done trace=%s scope=%s chunks=%s top_score=%s outcome=%s elapsed_ms=%d",
            trace, len(req.scope), len(chunks),
            None if top_score is None else round(top_score, 3),
            outcome, (time.monotonic() - started) * 1000,
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )