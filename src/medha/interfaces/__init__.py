"""Abstract base classes for embedders and storage backends."""

from medha.interfaces.embedder import BaseEmbedder
from medha.interfaces.storage import VectorStorageBackend

__all__ = ["BaseEmbedder", "VectorStorageBackend"]
