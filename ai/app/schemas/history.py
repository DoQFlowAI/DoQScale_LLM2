from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.base import CamelModel

Line = Annotated[str, Field(max_length=2000)]


class DiffBlock(CamelModel):
    type: Literal["INSERT", "DELETE", "CHANGE"]
    before: list[Line] = Field(max_length=200)
    after: list[Line] = Field(max_length=200)

    @model_validator(mode="after")
    def not_empty(self):
        if not self.before and not self.after:
            raise ValueError("empty block")
        return self


class HistorySegment(CamelModel):
    # 'from'은 파이썬 예약어라 필드 이름은 from_, JSON에서는 "from"
    from_: int = Field(alias="from", ge=1)
    to: int = Field(ge=1)
    blocks: list[DiffBlock] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def to_after_from(self):
        # 중간 버전이 삭제되면 1→3 처럼 번호가 건너뛸 수 있어서 '바로 다음'은 강제하지 않음
        if self.to <= self.from_:
            raise ValueError("to must be greater than from")
        return self


class HistoryRequest(CamelModel):
    document_id: int = Field(gt=0)
    segments: list[HistorySegment] = Field(min_length=1, max_length=20)


class HistorySegmentResult(CamelModel):
    from_: int = Field(alias="from")
    to: int
    change_type: Literal["PROCEDURE", "NUMERIC", "WORDING"] | None
    summary: str | None
    summary_status: Literal["OK", "FAILED"]


class HistoryResponse(CamelModel):
    segments: list[HistorySegmentResult]