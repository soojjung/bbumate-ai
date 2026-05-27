"""OpenAI 임베딩 모듈."""

import os

from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 객체를 생성하여 반환합니다.

    환경 변수에서 API 키를 자동 로드합니다.

    Returns:
        초기화된 OpenAIEmbeddings 객체.

    Raises:
        ValueError: 환경 변수가 설정되지 않은 경우.
    """
    openai_model = os.getenv("OPENAI_EMBEDDING_MODEL")

    if not os.getenv("OPENAI_API_KEY") or not openai_model:
        raise ValueError(
            "환경 변수 (OPENAI_API_KEY 또는 OPENAI_EMBEDDING_MODEL)가 설정되지 않았습니다. "
            "`.env` 파일을 확인하세요."
        )

    return OpenAIEmbeddings(model=openai_model)
