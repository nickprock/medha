"""Medha - Semantic Memory for AI Text-to-Query systems."""

__version__ = "0.3.0"

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.interfaces.embedder import BaseEmbedder
from medha.interfaces.l1_cache import L1CacheBackend
from medha.interfaces.storage import VectorStorageBackend
from medha.l1_cache.memory import InMemoryL1Cache
from medha.l1_cache.redis_adapter import RedisL1Cache
from medha.logging import setup_logging
from medha.types import CacheEntry, CacheHit, CacheResult, QueryTemplate, SearchStrategy

_optional: list[str] = []

try:
    from medha.backends.qdrant import QdrantBackend
    _optional.append("QdrantBackend")
except ImportError:
    pass

try:
    from medha.backends.pgvector import PgVectorBackend
    _optional.append("PgVectorBackend")
except ImportError:
    pass

__all__ = [
    "Medha", "Settings", "CacheHit", "QueryTemplate", "CacheEntry",
    "CacheResult", "SearchStrategy", "BaseEmbedder", "L1CacheBackend",
    "VectorStorageBackend", "InMemoryL1Cache", "RedisL1Cache", "setup_logging",
    "InMemoryBackend",
] + _optional
