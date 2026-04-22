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

try:
    from medha.backends.vectorchord import VectorChordBackend
    _all_extra.append("VectorChordBackend")
except ImportError:
    pass

try:
    from medha.backends.chroma import ChromaBackend
    _all_extra.append("ChromaBackend")
except ImportError:
    pass

try:
    from medha.backends.weaviate import WeaviateBackend
    _all_extra.append("WeaviateBackend")
except ImportError:
    pass

try:
    from medha.backends.redis_vector import RedisVectorBackend
    _all_extra.append("RedisVectorBackend")
except ImportError:
    pass

__all__ = ["InMemoryBackend"] + (["QdrantBackend"] if HAS_QDRANT else []) + _all_extra
