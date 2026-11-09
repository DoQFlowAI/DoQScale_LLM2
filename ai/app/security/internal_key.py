import hmac
import logging

from fastapi import Header, HTTPException, Request

from app.core.config import settings

logger = logging.getLogger("ai.security")


async def verify_internal_key(
    request: Request,
    x_internal_key: str | None = Header(default=None),
) -> None:
    """X-Internal-Key 헤더가 서버 설정값과 같은지 확인한다. 다르면 401."""
    expected = settings.AI_INTERNAL_KEY.get_secret_value().encode()
    provided = (x_internal_key or "").encode()

    # hmac.compare_digest: 앞 글자가 틀려도 끝까지 비교해서 걸리는 시간이 같다.
    # 일반 == 비교는 틀린 위치에서 바로 멈춰서, 응답 시간 차이로 키를 한 글자씩 추측당할 수 있다.
    if not hmac.compare_digest(provided, expected):
        # 키 값 자체는 로그에 남기지 않고, 헤더가 있었는지 여부만 남긴다
        logger.warning(
            "internal key rejected trace=%s path=%s header_present=%s",
            request.headers.get("X-Trace-Id", "-"),
            request.url.path,
            x_internal_key is not None,
        )
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")