import os
import tempfile

# config.py가 import되기 전에 테스트용 값을 고정한다.
# 환경변수가 .env보다 우선하므로, 개인 .env 설정과 무관하게 같은 결과가 나온다.
os.environ["AI_INTERNAL_KEY"] = "test-internal-key"
os.environ["ENABLE_DOCS"] = "false"

# 테스트가 실수로 진짜 OpenAI를 부르지 않게 한다:
# 가짜 키 + 아무것도 없는 주소 → 바꿔치기를 빠뜨린 테스트는 과금 없이 바로 실패한다
os.environ["LLM_PROVIDER"] = "openai"
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:9"
os.environ["GROQ_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"

# 개발용 ./faiss, ./uploads 를 건드리지 않도록 임시 폴더를 기본값으로
os.environ["VECTOR_STORE_DIR"] = tempfile.mkdtemp(prefix="ai-test-faiss-")
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="ai-test-uploads-")