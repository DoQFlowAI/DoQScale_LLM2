from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel


class ExtractRequest(CamelModel):
    version_id: int = Field(gt=0)
    file_path: str = Field(min_length=1, max_length=300)
    mime_type: Literal[
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # 텍스트 계열: Backend가 색인 직전에 임시 파일을 만들기 위해 호출한다
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/xml",
        "application/yaml",
    ]


class ExtractResponse(CamelModel):
    content_text: str
    page_count: int | None