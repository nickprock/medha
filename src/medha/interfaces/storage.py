"""VectorStorageBackend abstract class defining the storage interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from medha.types import CacheEntry, CacheResult


class VectorStorageBackend(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    async def initialize(self, collection_name: str, dimension: int, **kwargs) -> None:
        """Set up the storage backend (create collection, indexes, quantization).

        This method is idempotent: calling it twice with the same arguments
        must not raise or duplicate data.

        Args:
            collection_name: Name of the vector collection.
            dimension: Vector dimensionality (must match the embedder).
            **kwargs: Backend-specific configuration (quantization, HNSW params, etc.).

        Raises:
            StorageInitializationError: If setup fails.
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[CacheResult]:
        """Search for similar vectors.

        Args:
            collection_name: Collection to search.
            vector: Query vector.
            limit: Max number of results.
            score_threshold: Minimum similarity score (0.0 - 1.0).

        Returns:
            List of CacheResult, sorted by descending score.

        Raises:
            StorageError: If the search fails.
        """
        ...

    @abstractmethod
    async def upsert(self, collection_name: str, entries: List[CacheEntry]) -> None:
        """Insert or update cache entries.

        Args:
            collection_name: Target collection.
            entries: List of CacheEntry objects to upsert.

        Raises:
            StorageError: If the upsert fails.
        """
        ...

    @abstractmethod
    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_vectors: bool = False,
    ) -> tuple[List[CacheResult], Optional[str]]:
        """Iterate over all points in a collection.

        Used by fuzzy search (Tier 4) and admin operations.

        Args:
            collection_name: Collection to scroll.
            limit: Batch size per scroll.
            offset: Pagination token from a previous scroll.
            with_vectors: Whether to include vectors in results.

        Returns:
            Tuple of (results, next_offset). next_offset is None when done.

        Raises:
            StorageError: If the scroll fails.
        """
        ...

    @abstractmethod
    async def count(self, collection_name: str) -> int:
        """Return the number of points in a collection.

        Raises:
            StorageError: If the count fails.
        """
        ...

    @abstractmethod
    async def delete(self, collection_name: str, ids: List[str]) -> None:
        """Delete points by ID.

        Args:
            collection_name: Target collection.
            ids: List of point IDs to delete.

        Raises:
            StorageError: If the delete fails.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources (close connections, etc.)."""
        ...

    # --- Context manager support ---

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
