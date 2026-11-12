# ---------- 빌드: 의존성 설치 ----------
FROM python:3.12-slim AS builder

# 로컬 `uv --version`과 같은 버전으로 맞춘다
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
# --frozen: uv.lock 그대로 설치 / --no-dev: pytest 같은 개발용 제외
RUN uv sync --frozen --no-dev --no-install-project

# ---------- 실행 이미지 ----------
FROM python:3.12-slim

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VECTOR_STORE_DIR=/app/faiss \
    UPLOAD_DIR=/app/uploads \
    ENABLE_DOCS=false \
    LANGSMITH_TRACING=false

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app ./app

# root가 아닌 사용자로 실행. 색인 폴더만 쓰기 가능, 업로드는 읽기 전용으로 마운트됨
RUN useradd --system --uid 1001 --no-create-home app \
    && mkdir -p /app/faiss /app/uploads \
    && chown app:app /app/faiss
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# worker 1개: 인덱스 파일을 한 프로세스만 쓰게 한다 (명세서 6장)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]