"""Unit tests for medha.interfaces.embedder.BaseEmbedder ABC."""

import pytest

from medha.interfaces.embedder import BaseEmbedder


class TestBaseEmbedderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_partial_implementation_fails(self):
        class PartialEmbedder(BaseEmbedder):
            @property
            def dimension(self):
                return 384

            @property
            def model_name(self):
                return "partial"

            # Missing aembed and aembed_batch

        with pytest.raises(TypeError):
            PartialEmbedder()


class TestMockEmbedder:
    def test_dimension(self, mock_embedder):
        assert mock_embedder.dimension == 384

    def test_model_name(self, mock_embedder):
        assert mock_embedder.model_name == "mock-embedder"

    async def test_aembed_returns_correct_length(self, mock_embedder):
        vec = await mock_embedder.aembed("test")
        assert len(vec) == 384

    async def test_aembed_deterministic(self, mock_embedder):
        v1 = await mock_embedder.aembed("same text")
        v2 = await mock_embedder.aembed("same text")
        assert v1 == v2

    async def test_aembed_unit_normalized(self, mock_embedder):
        vec = await mock_embedder.aembed("test query")
        magnitude = sum(v ** 2 for v in vec) ** 0.5
        assert abs(magnitude - 1.0) < 1e-6

    async def test_aembed_batch(self, mock_embedder):
        vecs = await mock_embedder.aembed_batch(["a", "b"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384
        assert len(vecs[1]) == 384

    async def test_different_inputs_different_vectors(self, mock_embedder):
        v1 = await mock_embedder.aembed("apples")
        v2 = await mock_embedder.aembed("database schema")
        assert v1 != v2


class TestSyncWrappers:
    def test_embed_sync(self, mock_embedder):
        vec = mock_embedder.embed("test sync")
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_sync(self, mock_embedder):
        vecs = mock_embedder.embed_batch(["a", "b"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384


class TestEmbeddingsInit:
    def test_get_fastembed_adapter(self):
        from medha.embeddings import get_fastembed_adapter
        cls = get_fastembed_adapter()
        assert cls.__name__ == "FastEmbedAdapter"

    def test_get_openai_adapter(self):
        from medha.embeddings import get_openai_adapter
        cls = get_openai_adapter()
        assert cls.__name__ == "OpenAIAdapter"

    def test_fastembed_adapter_raises_import_error_when_not_installed(self):
        """FastEmbedAdapter.__init__ raises ImportError when fastembed is unavailable."""
        import medha.embeddings.fastembed_adapter as mod
        from medha.embeddings.fastembed_adapter import FastEmbedAdapter

        original = mod._FASTEMBED_AVAILABLE
        try:
            mod._FASTEMBED_AVAILABLE = False
            with pytest.raises(ImportError, match="pip install medha"):
                FastEmbedAdapter()
        finally:
            mod._FASTEMBED_AVAILABLE = original

    def test_openai_adapter_raises_import_error_when_not_installed(self):
        """OpenAIAdapter.__init__ raises ImportError when openai is unavailable."""
        import medha.embeddings.openai_adapter as mod
        from medha.embeddings.openai_adapter import OpenAIAdapter

        original = mod._OPENAI_AVAILABLE
        try:
            mod._OPENAI_AVAILABLE = False
            with pytest.raises(ImportError, match="pip install medha"):
                OpenAIAdapter()
        finally:
            mod._OPENAI_AVAILABLE = original


class TestFastEmbedAdapterErrors:
    """Tests for EmbeddingError (replacing assert) in FastEmbedAdapter."""

    def _uninitialized(self):
        """Return a FastEmbedAdapter instance bypassing __init__."""
        from medha.embeddings.fastembed_adapter import FastEmbedAdapter
        return object.__new__(FastEmbedAdapter)

    def test_dimension_raises_embedding_error_when_not_initialized(self):
        from medha.exceptions import EmbeddingError
        instance = self._uninitialized()
        instance._dimension = None
        with pytest.raises(EmbeddingError, match="dimension"):
            _ = instance.dimension

    def test_embed_sync_raises_embedding_error_when_model_none(self):
        from medha.exceptions import EmbeddingError
        instance = self._uninitialized()
        instance._model = None
        with pytest.raises(EmbeddingError, match="not initialized"):
            instance._embed_sync("test")

    def test_embed_batch_sync_raises_embedding_error_when_model_none(self):
        from medha.exceptions import EmbeddingError
        instance = self._uninitialized()
        instance._model = None
        with pytest.raises(EmbeddingError, match="not initialized"):
            instance._embed_batch_sync(["test"])

    def test_no_assertion_error_on_dimension(self):
        """Verify AssertionError is NOT raised (assertions disabled with -O)."""
        from medha.exceptions import EmbeddingError
        instance = self._uninitialized()
        instance._dimension = None
        with pytest.raises(EmbeddingError):
            _ = instance.dimension


class TestOpenAIAdapterErrors:
    """Tests for EmbeddingError (replacing assert) in OpenAIAdapter."""

    def _uninitialized(self):
        """Return an OpenAIAdapter instance bypassing __init__."""
        from medha.embeddings.openai_adapter import OpenAIAdapter
        return object.__new__(OpenAIAdapter)

    async def test_aembed_raises_embedding_error_when_client_none(self):
        from medha.exceptions import EmbeddingError
        instance = self._uninitialized()
        instance._client = None
        with pytest.raises(EmbeddingError, match="not initialized"):
            await instance.aembed("test")

    async def test_aembed_batch_raises_embedding_error_when_client_none(self):
        from medha.exceptions import EmbeddingError
        instance = self._uninitialized()
        instance._client = None
        with pytest.raises(EmbeddingError, match="not initialized"):
            await instance.aembed_batch(["test"])
