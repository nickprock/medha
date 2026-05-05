"""Google Gemini embeddings adapter implementing the BaseEmbedder interface."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    genai = None  # type: ignore[assignment]
    HAS_GEMINI = False

from medha.exceptions import ConfigurationError, EmbeddingError
from medha.interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)

_GEMINI_CHUNK_SIZE = 100


class GeminiAdapter(BaseEmbedder):
    """Embedding adapter using the Google Gemini Embedding API.

    Args:
        api_key: Google AI API key.
        model: Gemini embedding model. Defaults to "models/text-embedding-004".
        task_type_query: Task type for query embeddings.
        task_type_document: Task type for document embeddings.
        output_dimensionality: Optional dimension truncation.

    Raises:
        ConfigurationError: If the google-generativeai package is not installed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
        task_type_query: str = "RETRIEVAL_QUERY",
        task_type_document: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int | None = None,
    ) -> None:
        if not HAS_GEMINI:
            raise ConfigurationError(
                "google-generativeai is required for GeminiAdapter. "
                "Install it with: pip install medha[gemini]"
            )
        self._model = model
        self._task_type_query = task_type_query
        self._task_type_document = task_type_document
        self._output_dimensionality = output_dimensionality
        self._dimension: int | None = None
        genai.configure(api_key=api_key)
        logger.info("GeminiAdapter initialized with model '%s'", model)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                "Dimension not available. Call aembed() or aembed_batch() first."
            )
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def aembed(self, text: str) -> list[float]:
        """Generate a query embedding via Gemini API (via thread to avoid blocking)."""
        try:
            logger.debug("GeminiAdapter aembed: text_len=%d", len(text))
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "content": text,
                "task_type": self._task_type_query,
            }
            if self._output_dimensionality is not None:
                call_kwargs["output_dimensionality"] = self._output_dimensionality
            result = await asyncio.to_thread(genai.embed_content, **call_kwargs)
            vector = list(result["embedding"])
            self._dimension = len(vector)
            logger.debug("GeminiAdapter aembed: done, dim=%d", self._dimension)
            return vector
        except (ConfigurationError, RuntimeError):
            raise
        except Exception as e:
            logger.error("GeminiAdapter aembed failed: %s", e)
            raise EmbeddingError(f"Gemini aembed failed: {e}") from e

    async def aembed_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings for multiple texts in chunks of 100.

        Args:
            texts: List of texts to embed.
            **kwargs: Pass ``is_document=True`` to use the document task type.
        """
        is_document: bool = kwargs.get("is_document", False)
        task_type = self._task_type_document if is_document else self._task_type_query
        try:
            logger.debug(
                "GeminiAdapter aembed_batch: %d texts, is_document=%s", len(texts), is_document
            )
            chunks = [
                texts[i : i + _GEMINI_CHUNK_SIZE]
                for i in range(0, len(texts), _GEMINI_CHUNK_SIZE)
            ]

            async def _embed_chunk(chunk: list[str]) -> list[list[float]]:
                call_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "content": chunk,
                    "task_type": task_type,
                }
                if self._output_dimensionality is not None:
                    call_kwargs["output_dimensionality"] = self._output_dimensionality
                result = await asyncio.to_thread(genai.embed_content, **call_kwargs)
                return [list(v) for v in result["embedding"]]

            chunk_results: list[list[list[float]]] = await asyncio.gather(
                *[_embed_chunk(chunk) for chunk in chunks]
            )
            vectors = [vec for chunk_vecs in chunk_results for vec in chunk_vecs]
            if vectors:
                self._dimension = len(vectors[0])
            logger.debug("GeminiAdapter aembed_batch: done, %d vectors", len(vectors))
            return vectors
        except (ConfigurationError, RuntimeError):
            raise
        except Exception as e:
            logger.error("GeminiAdapter aembed_batch failed: %s", e)
            raise EmbeddingError(f"Gemini aembed_batch failed: {e}") from e
