"""Unit tests for GeminiAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import medha.embeddings.gemini_adapter as gemini_mod
from medha.embeddings.gemini_adapter import GeminiAdapter, _GEMINI_CHUNK_SIZE
from medha.exceptions import ConfigurationError


def _make_embed_result(vectors: list[list[float]]) -> dict:
    return {"embedding": vectors}


def _single_embed_result(vector: list[float]) -> dict:
    return {"embedding": vector}


def _mock_genai(embed_side_effect=None, embed_return=None) -> MagicMock:
    """Return a mock genai module."""
    mock = MagicMock()
    if embed_side_effect is not None:
        mock.embed_content.side_effect = embed_side_effect
    elif embed_return is not None:
        mock.embed_content.return_value = embed_return
    return mock


class TestGeminiAdapterImportGuard:
    def test_raises_configuration_error_when_gemini_missing(self):
        original = gemini_mod.HAS_GEMINI
        try:
            gemini_mod.HAS_GEMINI = False
            with pytest.raises(ConfigurationError, match="pip install medha"):
                GeminiAdapter(api_key="test")
        finally:
            gemini_mod.HAS_GEMINI = original


class TestGeminiAdapterDimension:
    def test_dimension_raises_before_any_embed_call(self):
        mock_g = _mock_genai()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(api_key="test")
                with pytest.raises(RuntimeError, match="Dimension not available"):
                    _ = adapter.dimension

    async def test_dimension_set_after_aembed(self):
        mock_g = _mock_genai(embed_return=_single_embed_result([0.1, 0.2, 0.3]))
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(api_key="test")
                await adapter.aembed("hello")
                assert adapter.dimension == 3


class TestGeminiAdapterAembed:
    async def test_aembed_uses_query_task_type(self):
        mock_g = _mock_genai(embed_return=_single_embed_result([0.1, 0.2]))
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(api_key="test", task_type_query="RETRIEVAL_QUERY")
                result = await adapter.aembed("test query")

        mock_g.embed_content.assert_called_once()
        call_kwargs = mock_g.embed_content.call_args.kwargs
        assert call_kwargs["task_type"] == "RETRIEVAL_QUERY"
        assert result == [0.1, 0.2]


class TestGeminiAdapterAembedBatch:
    async def test_aembed_batch_uses_query_task_type_by_default(self):
        mock_g = _mock_genai(embed_return=_make_embed_result([[0.1], [0.2]]))
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(api_key="test", task_type_query="RETRIEVAL_QUERY")
                await adapter.aembed_batch(["a", "b"])

        call_kwargs = mock_g.embed_content.call_args.kwargs
        assert call_kwargs["task_type"] == "RETRIEVAL_QUERY"

    async def test_aembed_batch_uses_document_task_type_when_is_document_true(self):
        mock_g = _mock_genai(embed_return=_make_embed_result([[0.1], [0.2]]))
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(
                    api_key="test",
                    task_type_query="RETRIEVAL_QUERY",
                    task_type_document="RETRIEVAL_DOCUMENT",
                )
                await adapter.aembed_batch(["a", "b"], is_document=True)

        call_kwargs = mock_g.embed_content.call_args.kwargs
        assert call_kwargs["task_type"] == "RETRIEVAL_DOCUMENT"

    async def test_aembed_batch_chunks_large_input(self):
        """Verify input >100 texts is split into multiple API calls."""
        total = _GEMINI_CHUNK_SIZE + 50  # 150 texts
        texts = [f"text {i}" for i in range(total)]
        call_count = 0
        expected_chunks = [_GEMINI_CHUNK_SIZE, 50]

        def fake_embed(**kwargs):
            nonlocal call_count
            n = len(kwargs["content"])
            assert n == expected_chunks[call_count], (
                f"chunk {call_count}: expected {expected_chunks[call_count]} texts, got {n}"
            )
            call_count += 1
            return _make_embed_result([[0.1] * 4 for _ in range(n)])

        mock_g = _mock_genai(embed_side_effect=fake_embed)
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(api_key="test")
                result = await adapter.aembed_batch(texts)

        assert call_count == 2
        assert len(result) == total

    async def test_aembed_batch_returns_flattened_results(self):
        mock_g = _mock_genai(embed_return=_make_embed_result([[0.1, 0.2], [0.3, 0.4]]))
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(api_key="test")
                result = await adapter.aembed_batch(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
