from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel


class ExtractRequest(CamelModel):
    version_id: int = Field(gt=0)
    file_path: str = Field(min_length=1, max_length=300)
    mime_type: Literal[
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]


class ExtractResponse(CamelModel):
    content_text: str
    page_count: int | None