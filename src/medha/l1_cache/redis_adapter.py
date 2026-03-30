"""Redis-backed L1 cache for distributed deployments.

Requires: ``pip install medha[redis]``
"""

from __future__ import annotations

import json
import logging

from medha.interfaces.l1_cache import L1CacheBackend
from medha.types import CacheHit, SearchStrategy

logger = logging.getLogger(__name__)


class RedisL1Cache(L1CacheBackend):
    """Redis-backed L1 cache.

    Enables sharing the L1 cache across multiple service instances
    (horizontal scaling).  Each entry is stored as a JSON-serialised
    ``CacheHit`` with an optional TTL.  LRU eviction is delegated to Redis
    — configure ``maxmemory-policy allkeys-lru`` on the Redis server for
    automatic eviction when memory is full.

    Args:
        url:      Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
        prefix:   Key namespace prefix (default: ``"medha:l1"``).
        ttl:      Optional entry TTL in seconds.  ``None`` = no expiry.
        max_size: Soft local size hint used for statistics.  Eviction is
                  handled by Redis, not by this adapter.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "medha:l1",
        ttl: int | None = None,
        max_size: int = 1000,
    ) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ImportError(
                "RedisL1Cache requires the 'redis' package. "
                "Install it with: pip install 'medha[redis]'"
            ) from exc

        self._prefix = prefix
        self._ttl = ttl
        self._max_size = max_size
        self._size_hint = 0  # Approximate; not decremented on Redis-side eviction
        self._client = aioredis.from_url(url, decode_responses=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def _serialise(self, hit: CacheHit) -> str:
        return json.dumps({
            "generated_query": hit.generated_query,
            "response_summary": hit.response_summary,
            "confidence": hit.confidence,
            "strategy": hit.strategy.value if hit.strategy else None,
            "template_used": hit.template_used,
        })

    def _deserialise(self, data: str) -> CacheHit:
        payload = json.loads(data)
        return CacheHit(
            generated_query=payload.get("generated_query"),
            response_summary=payload.get("response_summary"),
            confidence=payload.get("confidence", 1.0),
            strategy=SearchStrategy(payload["strategy"]) if payload.get("strategy") else SearchStrategy.NO_MATCH,
            template_used=payload.get("template_used"),
        )

    # ------------------------------------------------------------------
    # L1CacheBackend interface
    # ------------------------------------------------------------------

    async def get(self, key: str) -> CacheHit | None:
        try:
            data = await self._client.get(self._key(key))
            if data is None:
                return None
            return self._deserialise(data)
        except Exception as exc:
            logger.warning("RedisL1Cache.get failed (key=%s…): %s", key[:8], exc)
            return None

    async def set(self, key: str, value: CacheHit) -> None:
        try:
            payload = self._serialise(value)
            rkey = self._key(key)
            if self._ttl:
                await self._client.setex(rkey, self._ttl, payload)
            else:
                await self._client.set(rkey, payload)
            self._size_hint += 1
        except Exception as exc:
            logger.warning("RedisL1Cache.set failed (key=%s…): %s", key[:8], exc)

    async def clear(self) -> None:
        try:
            keys = await self._client.keys(f"{self._prefix}:*")
            if keys:
                await self._client.delete(*keys)
            self._size_hint = 0
        except Exception as exc:
            logger.warning("RedisL1Cache.clear failed: %s", exc)

    @property
    def size(self) -> int:
        """Local size hint — not decremented when Redis evicts entries."""
        return self._size_hint

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._client.aclose()
