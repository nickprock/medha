"""Integration tests for FastEmbedAdapter with real FastEmbed model."""

import pytest

fastembed = pytest.importorskip("fastembed")

from medha.embeddings.fastembed_adapter import FastEmbedAdapter


@pytest.fixture
def adapter():
    return FastEmbedAdapter()  # Default model: all-MiniLM-L6-v2


class TestFastEmbedAdapter:
    def test_dimension(self, adapter):
        assert adapter.dimension == 384

    def test_model_name(self, adapter):
        assert adapter.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    async def test_embed_single(self, adapter):
        vec = await adapter.aembed("test query")
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    async def test_embed_batch(self, adapter):
        vecs = await adapter.aembed_batch(["query one", "query two"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384
        assert len(vecs[1]) == 384

    async def test_deterministic(self, adapter):
        v1 = await adapter.aembed("same text")
        v2 = await adapter.aembed("same text")
        assert v1 == v2

    async def test_different_texts(self, adapter):
        v1 = await adapter.aembed("apples and oranges")
        v2 = await adapter.aembed("database schema design")
        # Cosine similarity should be low for unrelated texts
        dot = sum(a * b for a, b in zip(v1, v2))
        assert dot < 0.8
