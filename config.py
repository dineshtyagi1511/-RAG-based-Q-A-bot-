"""
config.py

Centralized application configuration using Pydantic Settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- OpenAI ----------------
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536

    # ---------------- Qdrant ----------------
    QDRANT_URL: str
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "rag_knowledge_base"

    # ---------------- Redis ----------------
    REDIS_URL: str
    REDIS_TTL: int = 3600

    # ---------------- Security ----------------
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW: int = 60
    MAX_QUERY_LENGTH: int = 500
    ENABLE_LLM_GUARD: bool = False

    # ---------------- RAG ----------------
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# ============================================================
# Backward Compatibility
# ============================================================

OPENAI_API_KEY = settings.OPENAI_API_KEY
LLM_MODEL = settings.LLM_MODEL
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DIM = settings.EMBEDDING_DIM

QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_COLLECTION = settings.QDRANT_COLLECTION

REDIS_URL = settings.REDIS_URL
REDIS_TTL = settings.REDIS_TTL

RATE_LIMIT_REQUESTS = settings.RATE_LIMIT_REQUESTS
RATE_LIMIT_WINDOW = settings.RATE_LIMIT_WINDOW
MAX_QUERY_LENGTH = settings.MAX_QUERY_LENGTH
ENABLE_LLM_GUARD = settings.ENABLE_LLM_GUARD

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
TOP_K = settings.TOP_K