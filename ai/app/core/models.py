"""LLM·임베딩 공급자 선택. 다른 코드는 get_llm(), get_embeddings()만 호출한다."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings

# 명세서 3.3: LLM 호출 타임아웃 30초, 재시도 1회
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 1


def _require_key(name: str, value: str) -> str:
    # 키가 비어 있으면 외부 호출 단계까지 가지 않고 바로 알 수 있게 한다
    if not value:
        raise RuntimeError(f"{name} 가 설정되지 않았습니다")
    return value


def get_llm(streaming: bool = False) -> BaseChatModel:
    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=_require_key("OPENAI_API_KEY", settings.OPENAI_API_KEY.get_secret_value()),
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
            streaming=streaming,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.LLM_MODEL,
            api_key=_require_key("GROQ_API_KEY", settings.GROQ_API_KEY.get_secret_value()),
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
            streaming=streaming,
        )

    raise RuntimeError("LLM_PROVIDER 값이 올바르지 않습니다 (openai | groq)")


def get_embeddings() -> Embeddings:
    provider = settings.EMBEDDING_PROVIDER

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=_require_key("OPENAI_API_KEY", settings.OPENAI_API_KEY.get_secret_value()),
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

    raise RuntimeError("EMBEDDING_PROVIDER 값이 올바르지 않습니다 (openai)")