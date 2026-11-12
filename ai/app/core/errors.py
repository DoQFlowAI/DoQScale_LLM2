class InvalidFileError(Exception):
    """파일 경로·형식·내용 문제. API에서 400 INVALID_FILE 로 바꾼다.

    메시지에는 고정 문구만 넣는다 (요청으로 들어온 경로 문자열을 넣지 않음).
    """


class AIUnavailableError(Exception):
    """LLM·임베딩 호출 실패, 인덱스 사용 불가. API에서 502 AI_UNAVAILABLE 로 바꾼다.""" 