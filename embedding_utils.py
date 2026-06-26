"""
embedding_utils.py

Singleton OpenAI Embeddings client.
"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Returns a singleton embedding model.
    """

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )