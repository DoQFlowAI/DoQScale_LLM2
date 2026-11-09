import os

# config.py가 import되기 전에 테스트용 값을 고정한다.
# 환경변수가 .env보다 우선하므로, 개인 .env 설정과 무관하게 같은 결과가 나온다.
os.environ["AI_INTERNAL_KEY"] = "test-internal-key"
os.environ["ENABLE_DOCS"] = "false"