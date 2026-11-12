from fastapi import APIRouter, Depends

from app.api import chat, extract, history, index
from app.security.internal_key import verify_internal_key

# /internal/** API는 전부 이 라우터 아래에 붙인다 → 키 검사가 자동 적용된다.
internal_router = APIRouter(prefix="/internal", dependencies=[Depends(verify_internal_key)])

# 새 API를 만들면 여기에 한 줄씩 추가 (main.py가 include 하기 전에 붙어 있어야 함)
internal_router.include_router(extract.router)
internal_router.include_router(index.router)
internal_router.include_router(chat.router)
internal_router.include_router(history.router)