"""Abstract base class for L1 cache backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from medha.types import CacheHit


class L1CacheBackend(ABC):
    """Interface for L1 (fast-lookup) cache backends.

    L1 cache sits in front of the vector backend and provides sub-millisecond
    responses for recently seen questions.  The default implementation is
    in-memory (``InMemoryL1Cache``); a Redis-backed implementation
    (``RedisL1Cache``) enables sharing the cache across multiple service
    instances in a horizontally-scaled deployment.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheHit]:
        """Return the cached hit for *key*, or ``None`` on a miss."""
        ...

    @abstractmethod
    async def set(self, key: str, value: CacheHit) -> None:
        """Store *value* under *key*.  Implementations handle eviction internally."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries from the cache."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Current number of entries.  May be approximate for distributed backends."""
        ...
