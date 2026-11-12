"""FAISS 인덱스 보관·저장·삭제 (명세서 6장).

- 인덱스 경로: {VECTOR_STORE_DIR}/{EMBEDDING_MODEL}/  → 모델이 바뀌면 다른 폴더
- 점수: 벡터를 길이 1로 정규화하고 내적으로 비교 → 점수 = 코사인 유사도 (-1~1, 높을수록 비슷)
- 쓰기(추가·삭제·저장)는 잠금 하나로 직렬화. uvicorn worker는 1개로 띄운다.
"""

import logging
import threading
import warnings
from collections.abc import Callable
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.core import models
from app.core.config import settings
from app.core.errors import AIUnavailableError

logger = logging.getLogger("ai.vector_store")
INDEX_NAME = "index"

# langchain-community는 내적 + 정규화 조합에 "Normalizing L2 is not applicable" 경고를 띄우지만,
# 실제로는 추가·검색 때 정규화를 그대로 적용한다 (test_chat: 같은 문장 점수 = 1). 로그만 어지러워 끈다.
warnings.filterwarnings("ignore", message="Normalizing L2 is not applicable")

# 만들 때와 불러올 때 반드시 같은 옵션을 써야 한다 (저장 파일에는 이 설정이 들어가지 않음)
FAISS_OPTIONS = {
    "distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT,
    "normalize_L2": True,
}


class VectorStore:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """메모리 상태를 비운다 (테스트에서 사용)."""
        self._store: FAISS | None = None
        self._loaded = False
        self.failed = False
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return Path(settings.VECTOR_STORE_DIR) / settings.EMBEDDING_MODEL

    # ---------- 불러오기 ----------

    def load(self) -> None:
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        self._loaded = True
        self.failed = False
        if not (self.directory / f"{INDEX_NAME}.faiss").exists():
            # 처음 기동: 빈 상태로 시작하고, 첫 색인 때 만든다
            self._store = None
            logger.info("vector store empty dir=%s", self.directory)
            return
        try:
            # allow_dangerous_deserialization: 인덱스 메타데이터가 pickle 형식이라 필요한 옵션.
            # 이 폴더는 AI Module만 쓰고 외부에 노출되지 않는 볼륨이라 우리가 만든 파일만 읽는다.
            self._store = FAISS.load_local(
                str(self.directory),
                models.get_embeddings(),
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
                **FAISS_OPTIONS,
            )
            logger.info("vector store loaded chunks=%s", len(self._store.docstore._dict))
        except Exception:
            self._store = None
            self.failed = True
            logger.exception("vector store load failed")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_unlocked()
        if self.failed:
            raise AIUnavailableError("vector store unavailable")

    # ---------- 쓰기 ----------

    def replace_version(self, docs: list[Document], ids: list[str], version_id: int) -> None:
        """한 버전의 조각을 (기존 것이 있으면 지우고) 새로 넣는다."""
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]

        with self._lock:
            self._ensure_loaded()
            embeddings = models.get_embeddings()

            # 1) 임베딩부터: 실패하면 저장소는 손대지 않은 채로 끝난다
            try:
                vectors = embeddings.embed_documents(texts)
            except Exception as e:
                raise AIUnavailableError("embedding failed") from e

            # 2) 같은 버전의 예전 조각 삭제 → 재색인해도 중복이 생기지 않음
            self._delete_unlocked(lambda md: md.get("versionId") == version_id)

            # 3) 추가 후 디스크에 저장
            pairs = list(zip(texts, vectors))
            if self._store is None:
                self._store = FAISS.from_embeddings(
                    pairs, embeddings, metadatas=metadatas, ids=ids, **FAISS_OPTIONS
                )
            else:
                self._store.add_embeddings(pairs, metadatas=metadatas, ids=ids)
            self._save_unlocked()

    def delete_where(self, predicate: Callable[[dict], bool]) -> int:
        with self._lock:
            self._ensure_loaded()
            deleted = self._delete_unlocked(predicate)
            if deleted:
                self._save_unlocked()
            return deleted

    def _delete_unlocked(self, predicate: Callable[[dict], bool]) -> int:
        if self._store is None:
            return 0
        # LangChain FAISS는 메타데이터로 지우는 기능이 없어서, 저장된 문서를 훑어 ID를 모은다
        ids = [doc_id for doc_id, doc in self._store.docstore._dict.items() if predicate(doc.metadata)]
        if ids:
            self._store.delete(ids)
        return len(ids)

    def _save_unlocked(self) -> None:
        if self._store is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.directory), index_name=INDEX_NAME)

    # ---------- 조회 ----------

    @property
    def store(self) -> FAISS | None:
        """검색용 (rag_chain 에서 사용)."""
        if not self._loaded:
            self.load()
        if self.failed:
            raise AIUnavailableError("vector store unavailable")
        return self._store

    def all_metadata(self) -> list[dict]:
        """저장된 조각들의 메타데이터 (테스트·점검용)."""
        if self._store is None:
            return []
        return [doc.metadata for doc in self._store.docstore._dict.values()]


vector_store = VectorStore()