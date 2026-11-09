from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """파이썬에서는 snake_case, JSON에서는 camelCase (versionId 등) 로 주고받는다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)