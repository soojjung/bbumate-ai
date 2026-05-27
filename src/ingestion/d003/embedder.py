"""3. Embed text"""

import os
from typing import Optional
from dotenv import load_dotenv

# OpenAI 임베딩 모델을 LangChain에서 사용할 수 있게 해주는 클래스
from langchain_openai import OpenAIEmbeddings


def get_openai_embeddings(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> OpenAIEmbeddings:
    """
    Initialize OpenAIEmbeddings with values from env when not explicitly provided:
    OpenAIEmbeddings를 초기화하고, .env 파일에서 값을 가져옴
    - api_key와 model_name이 제공되지 않으면, .env 파일에서 값을 가져옴. 제공되면 그 값을 사용.
    """
    load_dotenv()

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    resolved_model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL")

    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please configure .env")
    if not resolved_model:
        raise ValueError("OPENAI_EMBEDDING_MODEL is not set. Please configure .env")

    return OpenAIEmbeddings(api_key=resolved_api_key, model=resolved_model)
