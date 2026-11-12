# AI Module 배포 안내

AI Module(FastAPI)을 Compose에 붙일 때 필요한 내용입니다. 상세 API는 AI 모듈 연동 명세서(7번)를 봐주세요.

## 1. 이미지 빌드·배포

`main`에 push되면 GitHub Actions(`deploy.yml`)가 테스트 → 이미지 빌드 → 서버 배포를 합니다. (워크플로는 배포 담당 원종현님 작성)

- 이미지: `ghcr.io/doqflowai/doqflow-ai:{커밋 SHA}`, `:latest`
- 서버: `/opt/doqflow`의 공용 `docker-compose.yml`에서 `ai-module` 서비스만 교체 (`tags.env`의 `AI_TAG`)
- 확인: 컨테이너 `HEALTHCHECK`(Dockerfile)가 `healthy`가 될 때까지 대기
- PR(`develop`, `main` 대상)마다 `ci.yml`이 `uv run pytest` 실행

로컬에서 이미지만 확인할 때:

```bash
cd ai
docker build -t doqflow-ai .
```

## 2. Compose 설정 예시

```yaml
services:
  ai-module:
    build:
      context: ./DoQScale_LLM2/ai      # 배포 폴더 구조에 맞게 수정
    container_name: ai-module
    expose:
      - "8000"                         # ports 로 외부에 열지 않음 (Backend만 호출)
    environment:
      LLM_PROVIDER: openai
      LLM_MODEL: gpt-4o-mini
      EMBEDDING_PROVIDER: openai
      EMBEDDING_MODEL: text-embedding-3-small
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      AI_INTERNAL_KEY: ${AI_INTERNAL_KEY}
      RAG_SCORE_THRESHOLD: "0.20"
    volumes:
      - faiss_data:/app/faiss          # 색인 저장 (지우면 재색인 필요)
      - uploads:/app/uploads:ro        # Backend 업로드 볼륨을 읽기 전용으로
    restart: unless-stopped

  backend:
    # ... 기존 설정
    environment:
      AI_BASE_URL: http://ai-module:8000
      AI_INTERNAL_KEY: ${AI_INTERNAL_KEY}   # AI Module과 같은 값
    volumes:
      - uploads:/app/uploads           # Backend의 upload-dir 경로에 맞게
    depends_on:
      ai-module:
        condition: service_healthy

volumes:
  faiss_data:
  uploads:
```

## 3. 환경변수

| 이름 | 필수 | 값 | 비고 |
|------|------|-----|------|
| `OPENAI_API_KEY` | O | OpenAI 키 | GitHub Secrets |
| `AI_INTERNAL_KEY` | O | 16자 이상 임의 문자열 | GitHub Secrets. Backend와 같은 값. 없거나 짧으면 컨테이너가 기동하지 않음 |
| `LLM_PROVIDER` / `LLM_MODEL` | - | `openai` / `gpt-4o-mini` | 기본값 동일 |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | - | `openai` / `text-embedding-3-small` | 바꾸면 재색인 필요 |
| `RAG_SCORE_THRESHOLD` | - | `0.20` | 코사인 유사도 기준값. 질문셋 평가로 확정 (11/9, 11/10 재평가) |
| `VECTOR_STORE_DIR` / `UPLOAD_DIR` | - | `/app/faiss` / `/app/uploads` | 이미지 기본값. 바꾸지 않음 |
| `ENABLE_DOCS` | - | `false` | 배포에서는 Swagger 끔 |

`.env` 파일은 이미지에 들어가지 않습니다(`.dockerignore`). 값은 Compose 환경변수로만 주입합니다.

## 4. 주의사항

- **포트 비공개**: 8000은 `expose`만 합니다. Nginx도 `/internal/**`을 전달하지 않습니다.
- **업로드 볼륨 공유**: Backend가 `/internal/extract`에 보내는 `filePath`는 업로드 폴더 기준 상대 경로(`{documentId}/{versionNo}/{UUID}.pdf`)입니다. 두 컨테이너가 **같은 볼륨**을 봐야 합니다. AI는 uid 1001로 실행되므로 업로드 파일이 다른 사용자도 읽을 수 있는 권한(644, 폴더 755)이어야 합니다.
- **색인 볼륨**: `docker compose down -v`를 하면 `faiss_data`가 지워져 색인이 사라집니다. 필요하면 문서를 다시 업로드해 재색인합니다.
- **worker 1개**: 인덱스 파일을 한 프로세스만 쓰도록 고정되어 있습니다. 늘리지 않습니다.
- **첫 배포**: 배포 환경 색인은 비어 있습니다. 시연 문서를 다시 업로드해 색인합니다. 개발 PC의 `faiss/` 폴더를 서버로 복사하지 않습니다.
- **헬스체크**: `GET /health` → 200 `{"status":"ok"}`. 색인을 불러오지 못하면 503입니다.

## 5. Backend 타임아웃

- 연결 3초, 읽기 60초
- `/internal/history`만 읽기 180초
- Backend는 재시도하지 않습니다 (LLM 재시도는 AI Module이 1회)