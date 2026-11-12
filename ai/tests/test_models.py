import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.core import models
from app.core.config import settings


@pytest.fixture
def fake_keys(monkeypatch):
    # 테스트에서는 가짜 키를 넣는다 (실제 호출은 하지 않음)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("sk-test"))
    monkeypatch.setattr(settings, "GROQ_API_KEY", SecretStr("gsk-test"))


def test_openai_llm_selected(monkeypatch, fake_keys):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-4o-mini")
    llm = models.get_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"


def test_groq_llm_selected(monkeypatch, fake_keys):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_MODEL", "llama-3.3-70b-versatile")
    assert isinstance(models.get_llm(), ChatGroq)


def test_streaming_flag_passed(monkeypatch, fake_keys):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    assert models.get_llm(streaming=True).streaming is True


def test_unknown_llm_provider_fails(monkeypatch, fake_keys):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "something")
    with pytest.raises(RuntimeError):
        models.get_llm()


def test_missing_key_fails(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr(""))
    with pytest.raises(RuntimeError):
        models.get_llm()


def test_openai_embeddings_selected(monkeypatch, fake_keys):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
    emb = models.get_embeddings()
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.model == "text-embedding-3-small"


def test_unknown_embedding_provider_fails(monkeypatch, fake_keys):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "ollama")
    with pytest.raises(RuntimeError):
        models.get_embeddings()


def test_max_retries_override(monkeypatch, fake_keys):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    assert models.get_llm().max_retries == 1             # 기본: 클라이언트 재시도 1회
    assert models.get_llm(max_retries=0).max_retries == 0  # 변천사 그래프용: 끔