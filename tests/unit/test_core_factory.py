"""Unit tests for Medha._build_backend factory."""


import pytest

from medha.backends.memory import InMemoryBackend
from medha.backends.qdrant import QdrantBackend
from medha.config import Settings
from medha.core import Medha
from medha.exceptions import ConfigurationError


class TestBuildBackend:
    def test_factory_memory_backend(self, mock_embedder):
        settings = Settings(backend_type="memory")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, InMemoryBackend)

    def test_factory_qdrant_backend(self, mock_embedder):
        settings = Settings(backend_type="qdrant", qdrant_mode="memory")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, QdrantBackend)

    def test_factory_unknown_backend_raises(self, mock_embedder):
        settings = Settings(backend_type="memory")
        # Bypass Pydantic validation by directly patching the attribute
        object.__setattr__(settings, "backend_type", "nonexistent")
        with pytest.raises(ConfigurationError, match="Unknown backend_type"):
            Medha(collection_name="test", embedder=mock_embedder, settings=settings)

    def test_explicit_backend_skips_factory(self, mock_embedder):
        """When a backend is passed explicitly, _build_backend is never called."""
        settings = Settings(backend_type="qdrant")  # would build QdrantBackend
        explicit_backend = InMemoryBackend()
        m = Medha(
            collection_name="test",
            embedder=mock_embedder,
            backend=explicit_backend,
            settings=settings,
        )
        assert m._backend is explicit_backend

    def test_factory_pgvector_backend(self, mock_embedder):
        """backend_type='pgvector' instantiates PgVectorBackend (import tested, not connect)."""
        pytest.importorskip("asyncpg")
        from medha.backends.pgvector import PgVectorBackend

        settings = Settings(backend_type="pgvector")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, PgVectorBackend)

    def test_factory_vectorchord_backend(self, mock_embedder):
        pytest.importorskip("asyncpg")
        from medha.backends.vectorchord import VectorChordBackend

        settings = Settings(backend_type="vectorchord")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, VectorChordBackend)

    def test_factory_elasticsearch_backend(self, mock_embedder):
        pytest.importorskip("elasticsearch")
        from medha.backends.elasticsearch import ElasticsearchBackend

        settings = Settings(backend_type="elasticsearch")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, ElasticsearchBackend)

    def test_factory_chroma_backend(self, mock_embedder):
        pytest.importorskip("chromadb")
        from medha.backends.chroma import ChromaBackend

        settings = Settings(backend_type="chroma")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, ChromaBackend)

    def test_factory_weaviate_backend(self, mock_embedder):
        pytest.importorskip("weaviate")
        from medha.backends.weaviate import WeaviateBackend

        settings = Settings(backend_type="weaviate")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, WeaviateBackend)

    def test_factory_redis_backend(self, mock_embedder):
        pytest.importorskip("redis")
        from medha.backends.redis_vector import RedisVectorBackend

        settings = Settings(backend_type="redis")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, RedisVectorBackend)

    def test_factory_azure_search_backend(self, mock_embedder):
        pytest.importorskip("azure.search.documents")
        from medha.backends.azure_search import AzureSearchBackend

        settings = Settings(
            backend_type="azure-search",
            azure_search_endpoint="https://svc.search.windows.net",
            azure_search_api_key="key",
        )
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, AzureSearchBackend)
