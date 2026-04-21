"""Vector storage backend implementations."""

from medha.backends.memory import InMemoryBackend

try:
    from medha.backends.qdrant import QdrantBackend
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    from medha.backends.pgvector import PgVectorBackend
    _all_extra = ["PgVectorBackend"]
except ImportError:
    _all_extra = []

__all__ = ["InMemoryBackend"] + (["QdrantBackend"] if HAS_QDRANT else []) + _all_extra
