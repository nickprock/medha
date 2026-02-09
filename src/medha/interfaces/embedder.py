"""BaseEmbedder abstract class defining the embedder interface."""

from abc import ABC, abstractmethod
from typing import List
import asyncio


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension (e.g. 384, 768, 1536)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for logging/debugging."""
        ...

    @abstractmethod
    async def aembed(self, text: str) -> List[float]:
        """Generate an embedding for a single text string.

        Args:
            text: Input text to embed. Must be non-empty.

        Returns:
            A list of floats with length == self.dimension.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        ...

    @abstractmethod
    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of non-empty strings.

        Returns:
            A list of embeddings, one per input text, each with length == self.dimension.
            Order is preserved: result[i] corresponds to texts[i].

        Raises:
            EmbeddingError: If any embedding generation fails.
        """
        ...

    # --- Sync convenience wrappers ---

    def embed(self, text: str) -> List[float]:
        """Synchronous wrapper for aembed."""
        return self._run_sync(self.aembed(text))

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for aembed_batch."""
        return self._run_sync(self.aembed_batch(texts))

    @staticmethod
    def _run_sync(coro):
        """Run an async coroutine synchronously.

        Handles the case where an event loop is already running
        (e.g., inside Jupyter notebooks).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            # If a loop is running (Jupyter, etc.), use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
