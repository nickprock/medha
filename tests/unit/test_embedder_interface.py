"""Unit tests for medha.interfaces.embedder.BaseEmbedder ABC."""

import pytest

from medha.interfaces.embedder import BaseEmbedder
from tests.conftest import MockEmbedder


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
