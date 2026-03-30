"""InMemoryBackend — zero-external-deps vector storage backend."""

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any

from medha.exceptions import StorageError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity, using numpy if available."""
    try:
        import numpy as np

        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except ImportError:
        dot: float = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a: float = sum(x * x for x in a) ** 0.5
        norm_b: float = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


class InMemoryBackend(VectorStorageBackend):
    """Pure-Python in-process vector backend. No external dependencies required."""

    def __init__(self) -> None:
        # _store: collection_name -> {"dimension": int, "entries": {id: stored_point}}
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """No-op — backend is always connected."""

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if dimension <= 0:
            raise StorageError(f"dimension must be > 0, got {dimension}")
        async with self._lock:
            if collection_name not in self._store:
                self._store[collection_name] = {"dimension": dimension, "entries": {}}

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")

        entries = self._store[collection_name]["entries"]
        if not entries:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for point in entries.values():
            score = _cosine_similarity(vector, point["vector"])
            score = max(0.0, min(1.0, score))
            if score >= score_threshold:
                scored.append((score, point))

        scored.sort(key=lambda t: t[0], reverse=True)

        return [
            _point_to_cache_result(point, score)
            for score, point in scored[:limit]
        ]

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")
        if not entries:
            return
        async with self._lock:
            store_entries = self._store[collection_name]["entries"]
            for entry in entries:
                store_entries[entry.id] = {
                    "id": entry.id,
                    "vector": entry.vector,
                    "payload": {
                        "original_question": entry.original_question,
                        "normalized_question": entry.normalized_question,
                        "generated_query": entry.generated_query,
                        "query_hash": entry.query_hash,
                        "response_summary": entry.response_summary,
                        "template_id": entry.template_id,
                        "usage_count": entry.usage_count,
                        "created_at": entry.created_at.isoformat(),
                    },
                }

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")

        entries = self._store[collection_name]["entries"]
        ids = list(entries.keys())
        start = int(offset) if offset is not None else 0
        page_ids = ids[start : start + limit]
        next_offset = str(start + limit) if start + limit < len(ids) else None

        results = [
            _point_to_cache_result(entries[id_], score=1.0)
            for id_ in page_ids
        ]
        return results, next_offset

    async def count(self, collection_name: str) -> int:
        return len(self._store.get(collection_name, {}).get("entries", {}))

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        async with self._lock:
            entries = self._store.get(collection_name, {}).get("entries", {})
            for id_ in ids:
                entries.pop(id_, None)

    async def close(self) -> None:
        self._store.clear()

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")

        for point in self._store[collection_name]["entries"].values():
            if point["payload"]["query_hash"] == query_hash:
                return _point_to_cache_result(point, score=1.0)
        return None

    async def update_usage_count(self, collection_name: str, point_id: str) -> None:
        async with self._lock:
            entries = self._store.get(collection_name, {}).get("entries", {})
            if point_id not in entries:
                logger.warning(
                    "update_usage_count: id '%s' not found in collection '%s'",
                    point_id,
                    collection_name,
                )
                return
            entries[point_id]["payload"]["usage_count"] += 1


def _point_to_cache_result(point: dict[str, Any], score: float) -> CacheResult:
    payload = point["payload"]
    created_at_raw = payload.get("created_at")
    created_at: datetime | None = None
    if created_at_raw:
        with contextlib.suppress(ValueError, TypeError):
            created_at = datetime.fromisoformat(created_at_raw)
    return CacheResult(
        id=point["id"],
        score=max(0.0, min(1.0, score)),
        original_question=payload["original_question"],
        normalized_question=payload["normalized_question"],
        generated_query=payload["generated_query"],
        query_hash=payload["query_hash"],
        response_summary=payload.get("response_summary"),
        template_id=payload.get("template_id"),
        usage_count=payload.get("usage_count", 0),
        created_at=created_at,
    )
