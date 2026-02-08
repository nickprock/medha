"""FastEmbed adapter implementing the BaseEmbedder interface."""

import asyncio
import logging
from typing import List

try:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType
except ImportError:
    raise ImportError(
        "FastEmbed is required for FastEmbedAdapter. "
        "Install it with: pip install medha[fastembed]"
    )

from medha.exceptions import EmbeddingError
from medha.interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class FastEmbedAdapter(BaseEmbedder):
    """Embedding adapter using Onnx Runtime via FastEmbed.

    Supports any model available in the fastembed registry or
    custom HuggingFace models via ONNX export.

    Args:
        model_name: Model identifier. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        max_length: Maximum token length. Defaults to 512.
        cache_dir: Optional directory for model cache.

    Raises:
        EmbeddingError: If the model cannot be loaded.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        cache_dir: str | None = None,
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._dimension: int | None = None
        self._model: TextEmbedding | None = None

        self._load_model(model_name, max_length, cache_dir)

    @property
    def dimension(self) -> int:
        assert self._dimension is not None
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def aembed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        FastEmbed is synchronous under the hood; we run it in a thread
        to avoid blocking the event loop.
        """
        try:
            result = await asyncio.to_thread(self._embed_sync, text)
            return result
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed text with model '{self._model_name}': {e}"
            ) from e

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Uses fastembed's native batching for efficiency.
        """
        try:
            results = await asyncio.to_thread(self._embed_batch_sync, texts)
            return results
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed batch with model '{self._model_name}': {e}"
            ) from e

    def _embed_sync(self, text: str) -> List[float]:
        """Synchronous single-text embedding."""
        assert self._model is not None
        embeddings = list(self._model.embed([text]))
        vector = embeddings[0]
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding using native fastembed batching."""
        assert self._model is not None
        embeddings = list(self._model.embed(texts))
        return [
            vec.tolist() if hasattr(vec, "tolist") else list(vec)
            for vec in embeddings
        ]

    def _load_model(
        self, model_name: str, max_length: int, cache_dir: str | None
    ) -> None:
        """Load or register the embedding model.

        Logic:
            1. Check if model_name is in fastembed's supported models list.
            2. If yes, load directly.
            3. If no, attempt to register as a custom HuggingFace ONNX model.
            4. Probe dimension by embedding a test string.

        Raises:
            EmbeddingError: If model loading fails.
        """
        try:
            supported_models = [
                m["model"] for m in TextEmbedding.list_supported_models()
            ]

            if model_name in supported_models:
                logger.info("Using registered model: %s", model_name)
                self._model = TextEmbedding(
                    model_name=model_name,
                    max_length=max_length,
                    cache_dir=cache_dir,
                )
            else:
                logger.info("Registering custom model: %s", model_name)
                TextEmbedding.add_custom_model(
                    model=model_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=model_name),
                    model_file="onnx/model.onnx",
                )
                self._model = TextEmbedding(
                    model_name=model_name,
                    max_length=max_length,
                    cache_dir=cache_dir,
                )

            # Probe dimension by embedding a test string
            probe = list(self._model.embed(["dimension probe"]))
            self._dimension = len(probe[0])
            logger.info(
                "Model '%s' loaded, dimension=%d", model_name, self._dimension
            )

        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {e}"
            ) from e
