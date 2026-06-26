"""
cache.py — Redis-backed response cache for the RAG pipeline.

Cache key = SHA-256( collection_name + normalised_query )
Cache value = JSON blob  { answer, sources, model, chunks_retrieved }
TTL = REDIS_TTL seconds (default 1 hour)

Designed to be fail-soft: every method catches Redis errors and falls back
gracefully so the app continues to work without caching.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis

from config import REDIS_TTL, REDIS_URL

logger = logging.getLogger(__name__)


class RAGCache:
    """Thread-safe, fail-soft Redis cache for RAG query results."""

    def __init__(self) -> None:
        try:
            self._client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            self._client.ping()
            self.available = True
            logger.info("✅ Cache: Redis connected at %s", REDIS_URL)
        except Exception as exc:
            self.available = False
            logger.warning("⚠️  Cache: Redis unavailable (%s). Caching disabled.", exc)

    # ── Key helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(query: str, collection: str) -> str:
        """
        Deterministic, collision-resistant cache key.
        Normalise query to lowercase + stripped so minor case differences hit cache.
        """
        payload = f"{collection}||{query.lower().strip()}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"rag:{digest}"

    # ── Core operations ──────────────────────────────────────────────────────

    def get(self, query: str, collection: str) -> dict[str, Any] | None:
        """Return cached result dict, or None on miss / error."""
        if not self.available:
            return None
        try:
            raw = self._client.get(self._make_key(query, collection))
            if raw:
                logger.debug("🎯 Cache HIT  | q=%.60s", query)
                return json.loads(raw)
            logger.debug("💨 Cache MISS | q=%.60s", query)
            return None
        except Exception as exc:
            logger.error("Cache.get error: %s", exc)
            return None

    def set(
        self,
        query: str,
        collection: str,
        payload: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Persist payload; return True on success."""
        if not self.available:
            return False
        try:
            key = self._make_key(query, collection)
            self._client.setex(key, ttl or REDIS_TTL, json.dumps(payload, default=str))
            logger.debug("💾 Cache SET  | q=%.60s | ttl=%ds", query, ttl or REDIS_TTL)
            return True
        except Exception as exc:
            logger.error("Cache.set error: %s", exc)
            return False

    def invalidate_all(self) -> int:
        if not self.available:
            return 0

        deleted = 0

        try:
            cursor = 0

            while True:
                cursor, keys = self._client.scan(
                cursor=cursor,
                match="rag:*",
                count=100,
                )

                if keys:
                    deleted += self._client.delete(*keys)

                if cursor == 0:
                    break

            return deleted

        except Exception as exc:
            logger.error("Cache.invalidate error: %s", exc)
            return 0

    def invalidate_query(self, query: str, collection: str) -> bool:
        """Delete a single cached entry."""
        if not self.available:
            return False
        try:
            self._client.delete(self._make_key(query, collection))
            return True
        except Exception as exc:
            logger.error("Cache.invalidate_query error: %s", exc)
            return False

    # ── Observability ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return a dict suitable for display in the Streamlit sidebar."""
        if not self.available:
            return {"status": "unavailable", "cached_queries": 0}
        try:
            info = self._client.info("stats")
            cached = len(self._client.keys("rag:*"))
            return {
                "status": "connected",
                "cached_queries": cached,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "ttl_seconds": REDIS_TTL,
                "url": REDIS_URL,
            }
        except Exception as exc:
            return {"status": f"error: {exc}", "cached_queries": 0}


# Module-level singleton — shared across the app
cache = RAGCache()