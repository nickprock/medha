"""Cohere embeddings adapter implementing the BaseEmbedder interface."""

from __future__ import annotations

import logging
from typing import Any

try:
    import cohere
    HAS_COHERE = True
except ImportError:
    cohere = None  # type: ignore[assignment]
    HAS_COHERE = False

from medha.exceptions import ConfigurationError, EmbeddingError
from medha.interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class CohereAdapter(BaseEmbedder):
    """Embedding adapter using the Cohere Embed API v2.

    Args:
        api_key: Cohere API key.
        model: Cohere embedding model. Defaults to "embed-multilingual-v3.0".
        input_type_query: Input type used for query embeddings.
        input_type_document: Input type used for document embeddings.
        embedding_types: Optional list of embedding types to request.

    Raises:
        ConfigurationError: If the cohere package is not installed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "embed-multilingual-v3.0",
        input_type_query: str = "search_query",
        input_type_document: str = "search_document",
        embedding_types: list[str] | None = None,
    ) -> None:
        if not HAS_COHERE:
            raise ConfigurationError(
                "Cohere is required for CohereAdapter. "
                "Install it with: pip install medha[cohere]"
            )
        self._model = model
        self._input_type_query = input_type_query
        self._input_type_document = input_type_document
        self._embedding_types = embedding_types
        self._dimension: int | None = None
        self._client: cohere.AsyncClientV2 = cohere.AsyncClientV2(api_key=api_key)
        logger.info("CohereAdapter initialized with model '%s'", model)

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
        """Generate a query embedding via Cohere API."""
        try:
            logger.debug("CohereAdapter aembed: text_len=%d", len(text))
            kwargs: dict[str, Any] = {
                "texts": [text],
                "model": self._model,
                "input_type": self._input_type_query,
            }
            if self._embedding_types is not None:
                kwargs["embedding_types"] = self._embedding_types
            response = await self._client.embed(**kwargs)
            vector = list(response.embeddings.float_[0])
            self._dimension = len(vector)
            logger.debug("CohereAdapter aembed: done, dim=%d", self._dimension)
            return vector
        except (ConfigurationError, RuntimeError):
            raise
        except Exception as e:
            logger.error("CohereAdapter aembed failed: %s", e)
            raise EmbeddingError(f"Cohere aembed failed: {e}") from e

    async def aembed_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            **kwargs: Pass ``is_document=True`` to use the document input type.
        """
        is_document: bool = kwargs.get("is_document", False)
        input_type = self._input_type_document if is_document else self._input_type_query
        try:
            logger.debug(
                "CohereAdapter aembed_batch: %d texts, is_document=%s", len(texts), is_document
            )
            call_kwargs: dict[str, Any] = {
                "texts": texts,
                "model": self._model,
                "input_type": input_type,
            }
            if self._embedding_types is not None:
                call_kwargs["embedding_types"] = self._embedding_types
            response = await self._client.embed(**call_kwargs)
            vectors = [list(v) for v in response.embeddings.float_]
            if vectors:
                self._dimension = len(vectors[0])
            logger.debug("CohereAdapter aembed_batch: done, %d vectors", len(vectors))
            return vectors
        except (ConfigurationError, RuntimeError):
            raise
        except Exception as e:
            logger.error("CohereAdapter aembed_batch failed: %s", e)
            raise EmbeddingError(f"Cohere aembed_batch failed: {e}") from e
