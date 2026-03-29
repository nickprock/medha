"""Unit tests for Medha._build_backend factory."""

from unittest.mock import patch

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
        asyncpg = pytest.importorskip("asyncpg")
        from medha.backends.pgvector import PgVectorBackend

        settings = Settings(backend_type="pgvector")
        m = Medha(collection_name="test", embedder=mock_embedder, settings=settings)
        assert isinstance(m._backend, PgVectorBackend)
