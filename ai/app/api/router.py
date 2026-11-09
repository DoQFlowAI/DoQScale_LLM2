from fastapi import APIRouter, Depends

from app.security.internal_key import verify_internal_key

# /internal/** API는 전부 이 라우터 아래에 붙인다 → 키 검사가 자동 적용된다.
# 사용 예 (feat/extract 에서):
#   from app.api import extract
#   internal_router.include_router(extract.router)
internal_router = APIRouter(prefix="/internal", dependencies=[Depends(verify_internal_key)])