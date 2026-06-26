"""
qdrant_utils.py

Provides a singleton Qdrant client that is reused across the application.
"""

from functools import lru_cache

from qdrant_client import QdrantClient

from config import (
    QDRANT_API_KEY,
    QDRANT_URL,
)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Returns a singleton Qdrant client.

    The same client instance is reused across the application,
    improving connection reuse and reducing latency.
    """

    kwargs = {
        "url": QDRANT_URL,
        "timeout": 30,
        "check_compatibility": False,
    }

    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY

    return QdrantClient(**kwargs)