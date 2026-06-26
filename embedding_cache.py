from __future__ import annotations
import hashlib
from cache import cache

class EmbeddingCache:
    PREFIX = "embedding:"

    @staticmethod
    def _key(text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{EmbeddingCache.PREFIX}{digest}"

    @classmethod
    def get(cls, text: str):
        return cache.get_json(cls._key(text))

    @classmethod
    def set(cls, text: str, embedding: list[float]):
        # 86400 seconds = 24 hours
        return cache.set_json(cls._key(text), embedding, ttl=86400)