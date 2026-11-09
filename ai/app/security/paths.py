from pathlib import Path

from app.core.config import settings
from app.core.errors import InvalidFileError


def resolve_upload_path(relative_path: str) -> Path:
    """Backend가 보낸 상대 경로를 UPLOAD_DIR 기준 실제 경로로 바꾼다.

    UPLOAD_DIR 밖을 가리키거나 파일이 없으면 InvalidFileError.
    """
    if not relative_path or "\x00" in relative_path:
        raise InvalidFileError("empty path or null byte")

    candidate = Path(relative_path)
    # 절대 경로(/etc/passwd, C:\..., \\server\...)는 받지 않는다
    if candidate.is_absolute() or candidate.drive or candidate.anchor:
        raise InvalidFileError("absolute path")

    base = Path(settings.UPLOAD_DIR).resolve()
    # resolve(): ../ 와 심볼릭 링크를 모두 풀어서 '진짜 위치'를 구한다
    target = (base / candidate).resolve()

    if not target.is_relative_to(base):
        raise InvalidFileError("outside upload dir")
    if not target.is_file():
        raise InvalidFileError("not a file")
    return target