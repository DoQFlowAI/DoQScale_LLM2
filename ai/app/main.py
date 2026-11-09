import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ai")

app = FastAPI(
    title="DoQScale AI Module",
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

# 명세서 4.7: 응답에는 code만, 상세 원인은 서버 로그에만
KNOWN_CODES = {"VALIDATION_ERROR", "UNAUTHORIZED", "INVALID_FILE", "AI_UNAVAILABLE", "NOT_FOUND"}
STATUS_TO_CODE = {400: "VALIDATION_ERROR", 401: "UNAUTHORIZED", 404: "NOT_FOUND", 405: "NOT_FOUND"}


def _trace_id(request: Request) -> str:
    return request.headers.get("X-Trace-Id", "-")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # FastAPI 기본 422 → 400. 어떤 필드가 틀렸는지는 응답에 넣지 않음
    # 로그에도 입력값(질문 원문 등)은 남기지 않고 필드 위치만 남김
    fields = [".".join(str(p) for p in e.get("loc", [])) for e in exc.errors()]
    logger.warning("validation failed trace=%s path=%s fields=%s", _trace_id(request), request.url.path, fields)
    return JSONResponse(status_code=400, content={"code": "VALIDATION_ERROR"})


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    # 코드에서 raise HTTPException(400, detail="INVALID_FILE") 처럼 쓰면 그 code가 나감
    code = exc.detail if exc.detail in KNOWN_CODES else STATUS_TO_CODE.get(exc.status_code, "INTERNAL_ERROR")
    logger.info("http error trace=%s path=%s status=%s code=%s", _trace_id(request), request.url.path, exc.status_code, code)
    return JSONResponse(status_code=exc.status_code, content={"code": code})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    # 예상 못 한 오류: 스택은 로그에만, 응답은 고정 문구
    logger.exception("unhandled error trace=%s path=%s", _trace_id(request), request.url.path)
    return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR"})


@app.get("/health")
def health():
    # 인덱스 로딩을 붙이면(feat/index) 실패 시 503을 돌려주도록 바꿀 예정
    return {"status": "ok"}