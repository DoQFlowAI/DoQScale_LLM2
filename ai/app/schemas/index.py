from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import CamelModel

PositiveId = Annotated[int, Field(gt=0)]


class IndexRequest(CamelModel):
    document_id: PositiveId
    version_id: PositiveId
    version_no: PositiveId


class IndexResponse(CamelModel):
    status: Literal["READY"]
    chunk_count: int


class DeleteIndexRequest(CamelModel):
    document_id: PositiveId
    # 기본값 없음: 필드를 빠뜨리면 400. 문서 전체 삭제는 null을 명시해야만 한다
    keep_version_id: PositiveId | None


class DeleteIndexResponse(CamelModel):
    deleted: int