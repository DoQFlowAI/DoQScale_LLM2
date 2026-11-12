from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수(.env)에서 읽는 설정. 코드에 값을 직접 쓰지 않는다."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 모델 선택
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 800
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # 키: SecretStr이라 print·로그에 찍혀도 '**********'로 나옴
    OPENAI_API_KEY: SecretStr = SecretStr("")
    GROQ_API_KEY: SecretStr = SecretStr("")
    AI_INTERNAL_KEY: SecretStr  # 기본값 없음 → 없으면 서버가 아예 안 뜸 (fail-closed)

    # 경로
    VECTOR_STORE_DIR: str = "./faiss"
    UPLOAD_DIR: str = "./uploads"

    # RAG
    RAG_TOP_K: int = 5
    RAG_FETCH_K: int = 100
    RAG_SCORE_THRESHOLD: float = 0.20  # 질문셋 평가(evaluation/)로 확정
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # 개발 편의: true면 /docs(Swagger) 화면을 켬. 배포에서는 false
    ENABLE_DOCS: bool = False

    @field_validator("AI_INTERNAL_KEY")
    @classmethod
    def internal_key_long_enough(cls, v: SecretStr) -> SecretStr:
        # 짧은 키("1234" 같은)로는 아예 서버가 뜨지 않게 막는다
        if len(v.get_secret_value()) < 16:
            raise ValueError("AI_INTERNAL_KEY는 16자 이상이어야 합니다")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()