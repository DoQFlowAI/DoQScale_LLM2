# DoQFlowAI AI Module

사내 문서 버전관리·RAG 질의응답 시스템의 AI 모듈(FastAPI)입니다. Backend만 호출하는 내부 API로 문서 추출, 색인, 질의응답, 버전 변천사 요약을 맡습니다.
상세 규격은 **AI 모듈 연동 명세서(7번)**, 배포는 [DEPLOY.md](DEPLOY.md)를 봐주세요.

## 하는 일

| API | 역할 |
|-----|------|
| `GET /health` | 상태 확인. 색인을 불러오지 못하면 503 |
| `POST /internal/extract` | PDF·DOCX·텍스트 계열 파일에서 본문 추출, 문장 단위 정리 (PDF는 쪽번호·머리말 제거) |
| `POST /internal/index` | 버전 본문을 800/100 청크로 나눠 FAISS에 색인 (PDF는 쪽 번호 유지) |
| `DELETE /internal/index` | 문서·버전의 벡터 삭제 (`keepVersionId`로 최신 버전만 남김) |
| `POST /internal/chat` | 권한 scope 안에서만 검색해 SSE로 답변 + 근거 (근거 없으면 noAnswer) |
| `POST /internal/history` | LangGraph로 버전 구간별 변경 요약·유형 분류 |

`/health`를 뺀 모든 요청에 `X-Internal-Key`가 필요합니다.

## 기술 스택

FastAPI · LangChain · LangGraph · FAISS(코사인) · OpenAI `gpt-4o-mini` / `text-embedding-3-small` · uv · pytest

## 폴더 구조

```
ai/
├── app/
│   ├── api/          내부 API 라우터
│   ├── chains/       RAG 체인 (프롬프트, 근거 정리)
│   ├── graphs/       변천사 요약 LangGraph
│   ├── core/         설정, 모델, 추출, 색인, FAISS 보관
│   ├── schemas/      요청·응답 모델
│   └── security/     내부 키 검사, 경로 검증
├── tests/            pytest 95개 (외부 호출 없음)
├── evaluation/       질문셋 평가 스크립트, 질문 60개, 시연 문서 7개
├── Dockerfile
└── DEPLOY.md
```

## 로컬 실행 (Git Bash)

```bash
cd ai
uv sync
cp .env.example .env          # OPENAI_API_KEY, AI_INTERNAL_KEY(16자 이상) 채우기
uv run uvicorn app.main:app --reload --port 8000
```

Swagger(`/docs`)는 `.env`에 `ENABLE_DOCS=true`일 때만 열립니다.

## 테스트

```bash
uv run pytest
```

가짜 임베딩·가짜 LLM으로 돌아가며 OpenAI를 호출하지 않습니다. PR마다 CI에서도 실행됩니다.

## 질문셋 평가

```bash
uv run python evaluation/run_eval.py          # 실제 OpenAI 호출 (과금)
uv run python evaluation/run_eval.py --fake   # 흐름만 확인
```

결과는 `evaluation/results/`(커밋 제외)에 md·csv로 저장됩니다.

| 항목 | 목표 | 결과 (11/10) |
|------|------|------|
| 근거율 | 90% 이상 | 97% (37/38) |
| 근거 없음 | 100% | 10/10 |
| 권한 격리·Prompt Injection 누출 | 0건 | 0건 |
| PDF 근거 쪽번호 | 100% | 7/7 |
| 응답 시간 p95 | 15초 이내 | 3.0초 |

시연 문서와 변경 정답표는 [evaluation/docs/README.md](evaluation/docs/README.md)에 있습니다.

## 보안

- 내부 키는 `hmac.compare_digest`로 비교하고, 없거나 16자 미만이면 서버가 뜨지 않습니다.
- 파일 경로는 업로드 폴더 밖으로 나가면 거부합니다.
- 검색은 Backend가 준 scope(버전 ID) 안에서만 하고, 결과를 한 번 더 확인합니다. 벡터에는 권한 정보를 넣지 않습니다.
- 문서 본문은 태그로 감싸 지시문으로 실행되지 않게 합니다.
- 컨테이너는 root가 아닌 사용자(uid 1001)로 실행하고, 8000 포트는 외부에 열지 않습니다.
