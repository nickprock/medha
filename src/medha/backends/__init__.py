"""Vector storage backend implementations."""

from medha.backends.memory import InMemoryBackend

try:
    from medha.backends.qdrant import QdrantBackend
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

_all_extra: list[str] = []

try:
    from medha.backends.pgvector import PgVectorBackend
    _all_extra.append("PgVectorBackend")
except ImportError:
    pass

try:
    from medha.backends.elasticsearch import ElasticsearchBackend
    _all_extra.append("ElasticsearchBackend")
except ImportError:
    pass

__all__ = ["InMemoryBackend"] + (["QdrantBackend"] if HAS_QDRANT else []) + _all_extra
