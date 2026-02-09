"""Medha - Semantic Memory for AI Text-to-Query systems."""

__version__ = "0.1.0"

from medha.core import Medha
from medha.config import Settings
from medha.types import CacheHit, QueryTemplate, CacheEntry, CacheResult, SearchStrategy
from medha.interfaces.embedder import BaseEmbedder
from medha.interfaces.storage import VectorStorageBackend
from medha.logging import setup_logging

__all__ = [
    "Medha", "Settings", "CacheHit", "QueryTemplate", "CacheEntry",
    "CacheResult", "SearchStrategy", "BaseEmbedder", "VectorStorageBackend",
    "setup_logging",
]
