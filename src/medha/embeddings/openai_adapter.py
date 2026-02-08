"""OpenAI embeddings adapter implementing the BaseEmbedder interface."""

import logging
from typing import List

try:
    from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, RateLimitError
except ImportError:
    raise ImportError(
        "OpenAI is required for OpenAIAdapter. "
        "Install it with: pip install medha[openai]"
    )

from medha.exceptions import EmbeddingError
from medha.interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseEmbedder):
    """Embedding adapter using the OpenAI Embeddings API.

    Args:
        model_name: OpenAI model identifier. Defaults to "text-embedding-3-small".
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        dimensions: Optional dimension override (for models that support it).

    Raises:
        EmbeddingError: If the OpenAI client cannot be initialized.
    """

    _KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ):
        self._model_name = model_name
        self._dimensions = dimensions
        self._client: AsyncOpenAI | None = None

        self._initialize_client(api_key)

    @property
    def dimension(self) -> int:
        if self._dimensions:
            return self._dimensions
        dim = self._KNOWN_DIMENSIONS.get(self._model_name)
        if dim is None:
            raise EmbeddingError(
                f"Unknown dimension for model '{self._model_name}'. "
                "Pass 'dimensions' explicitly."
            )
        return dim

    @property
    def model_name(self) -> str:
        return self._model_name

    async def aembed(self, text: str) -> List[float]:
        """Generate embedding via OpenAI API.

        Uses the async client for non-blocking operation.
        """
        assert self._client is not None
        try:
            logger.debug("OpenAI aembed: text_len=%d, model='%s'", len(text), self._model_name)
            kwargs = {"input": text, "model": self._model_name}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = await self._client.embeddings.create(**kwargs)
            logger.debug("OpenAI aembed: done, dim=%d", len(response.data[0].embedding))
            return response.data[0].embedding
        except AuthenticationError as e:
            logger.error("OpenAI authentication failed: %s", e)
            raise EmbeddingError(
                f"OpenAI authentication failed: {e}"
            ) from e
        except RateLimitError as e:
            logger.warning("OpenAI rate limit exceeded: %s", e)
            raise EmbeddingError(
                f"OpenAI rate limit exceeded: {e}"
            ) from e
        except APIConnectionError as e:
            logger.error("OpenAI API connection error: %s", e)
            raise EmbeddingError(
                f"OpenAI API connection error: {e}"
            ) from e
        except Exception as e:
            logger.error("OpenAI aembed failed: %s", e)
            raise EmbeddingError(
                f"Failed to embed text with OpenAI model '{self._model_name}': {e}"
            ) from e

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch via OpenAI API.

        OpenAI's API natively supports batched input.
        Respects rate limits via the client's built-in retry logic.
        """
        assert self._client is not None
        try:
            logger.debug("OpenAI aembed_batch: %d texts, model='%s'", len(texts), self._model_name)
            kwargs = {"input": texts, "model": self._model_name}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = await self._client.embeddings.create(**kwargs)
            # Sort by index to preserve input order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            logger.debug("OpenAI aembed_batch: done, %d vectors", len(sorted_data))
            return [item.embedding for item in sorted_data]
        except AuthenticationError as e:
            logger.error("OpenAI authentication failed: %s", e)
            raise EmbeddingError(
                f"OpenAI authentication failed: {e}"
            ) from e
        except RateLimitError as e:
            logger.warning("OpenAI rate limit exceeded: %s", e)
            raise EmbeddingError(
                f"OpenAI rate limit exceeded: {e}"
            ) from e
        except APIConnectionError as e:
            logger.error("OpenAI API connection error: %s", e)
            raise EmbeddingError(
                f"OpenAI API connection error: {e}"
            ) from e
        except Exception as e:
            logger.error("OpenAI aembed_batch failed: %s", e)
            raise EmbeddingError(
                f"Failed to embed batch with OpenAI model '{self._model_name}': {e}"
            ) from e

    def _initialize_client(self, api_key: str | None) -> None:
        """Initialize the async OpenAI client.

        Raises:
            EmbeddingError: If initialization fails.
        """
        try:
            self._client = AsyncOpenAI(api_key=api_key)
            logger.info("OpenAI client initialized for model '%s'", self._model_name)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize OpenAI client: {e}"
            ) from e
