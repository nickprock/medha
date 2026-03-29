"""Medha - Semantic Memory for AI Text-to-Query systems."""

__version__ = "0.1.0"

from medha.core import Medha
from medha.config import Settings
from medha.types import CacheHit, QueryTemplate, CacheEntry, CacheResult, SearchStrategy
from medha.interfaces.embedder import BaseEmbedder
from medha.interfaces.l1_cache import L1CacheBackend
from medha.interfaces.storage import VectorStorageBackend
from medha.l1_cache.memory import InMemoryL1Cache
from medha.l1_cache.redis_adapter import RedisL1Cache
from medha.logging import setup_logging
from medha.backends.memory import InMemoryBackend

__all__ = [
    "Medha", "Settings", "CacheHit", "QueryTemplate", "CacheEntry",
    "CacheResult", "SearchStrategy", "BaseEmbedder", "L1CacheBackend",
    "VectorStorageBackend", "InMemoryL1Cache", "RedisL1Cache", "setup_logging",
    "InMemoryBackend",
]
