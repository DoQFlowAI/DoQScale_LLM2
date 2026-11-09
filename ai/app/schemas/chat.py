from typing import Annotated

from pydantic import Field, StrictInt, field_validator

from app.schemas.base import CamelModel

# StrictInt: "33"(문자열)이나 true 같은 값을 숫자로 바꿔주지 않고 거부한다
VersionId = Annotated[StrictInt, Field(gt=0)]


class ChatRequest(CamelModel):
    question: str = Field(min_length=1, max_length=1000)
    # 없거나 빈 배열이거나 형식이 틀리면 400 → 검색 자체를 하지 않음 (fail-closed)
    scope: list[VersionId] = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("blank question")
        return v