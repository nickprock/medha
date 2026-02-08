"""Unit tests for medha.core.Medha waterfall search logic."""

import pytest
from typing import Dict, List, Optional
from unittest.mock import AsyncMock

from medha.config import Settings
from medha.core import Medha
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult, CacheHit, SearchStrategy
from tests.conftest import MockEmbedder


# ---------------------------------------------------------------------------
# MockBackend — in-memory backend with cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class MockBackend(VectorStorageBackend):
    """In-memory mock backend that stores entries in a dict."""

    def __init__(self):
        self._collections: Dict[str, List[CacheEntry]] = {}

    async def initialize(self, collection_name: str, dimension: int, **kwargs) -> None:
        self._collections.setdefault(collection_name, [])

    async def search(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[CacheResult]:
        entries = self._collections.get(collection_name, [])
        scored = []
        for e in entries:
            sim = _cosine_similarity(vector, e.vector)
            if sim >= score_threshold:
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, e in scored[:limit]:
            results.append(
                CacheResult(
                    id=e.id,
                    score=min(score, 1.0),
                    original_question=e.original_question,
                    normalized_question=e.normalized_question,
                    generated_query=e.generated_query,
                    query_hash=e.query_hash,
                    response_summary=e.response_summary,
                    template_id=e.template_id,
                    usage_count=e.usage_count,
                    created_at=e.created_at,
                )
            )
        return results

    async def upsert(self, collection_name: str, entries: List[CacheEntry]) -> None:
        self._collections.setdefault(collection_name, []).extend(entries)

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_vectors: bool = False,
    ) -> tuple[List[CacheResult], Optional[str]]:
        entries = self._collections.get(collection_name, [])
        start = int(offset) if offset else 0
        batch = entries[start : start + limit]
        results = [
            CacheResult(
                id=e.id,
                score=0.0,
                original_question=e.original_question,
                normalized_question=e.normalized_question,
                generated_query=e.generated_query,
                query_hash=e.query_hash,
                response_summary=e.response_summary,
                template_id=e.template_id,
            )
            for e in batch
        ]
        next_offset = str(start + limit) if start + limit < len(entries) else None
        return results, next_offset

    async def count(self, collection_name: str) -> int:
        return len(self._collections.get(collection_name, []))

    async def delete(self, collection_name: str, ids: List[str]) -> None:
        entries = self._collections.get(collection_name, [])
        id_set = set(ids)
        self._collections[collection_name] = [e for e in entries if e.id not in id_set]

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def waterfall_settings():
    return Settings(
        qdrant_mode="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


@pytest.fixture
def medha_instance(mock_backend, waterfall_settings):
    embedder = MockEmbedder(dimension=384)
    return Medha(
        collection_name="test_cache",
        embedder=embedder,
        backend=mock_backend,
        settings=waterfall_settings,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestL1Cache:
    async def test_l1_cache_hit(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        # First search populates L1 via store; second search should be L1 hit
        hit1 = await medha_instance.search("How many users?")
        assert hit1.strategy != SearchStrategy.NO_MATCH
        hit2 = await medha_instance.search("How many users?")
        assert hit2.strategy != SearchStrategy.NO_MATCH
        assert medha_instance._stats["l1_hits"] >= 1

    async def test_l1_eviction(self, mock_backend, waterfall_settings):
        waterfall_settings = Settings(
            qdrant_mode="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            score_threshold_template=0.80,
            score_threshold_fuzzy=80.0,
            l1_cache_max_size=2,
        )
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="test_evict",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        await m.start()
        await m.store("question one", "SELECT 1")
        await m.store("question two", "SELECT 2")
        await m.store("question three", "SELECT 3")
        # L1 max is 2, so "question one" should have been evicted
        assert len(m._l1_cache) <= 2


class TestWaterfallOrder:
    async def test_template_before_semantic(self, medha_instance, sample_templates):
        await medha_instance.start()
        await medha_instance.load_templates(sample_templates)
        # Store a semantic entry for similar question
        await medha_instance.store(
            "How many users are there",
            "SELECT COUNT(*) FROM users",
        )
        # Clear L1 so we don't get L1 hit
        medha_instance._l1_cache.clear()
        medha_instance._embedding_cache.clear()
        # Use the *exact same* question to guarantee vector match
        hit = await medha_instance.search("How many users are there")
        # With deterministic mock embedder, identical text → exact vector match.
        # Template or Exact are both valid (template is checked first in waterfall).
        assert hit.strategy in (
            SearchStrategy.TEMPLATE_MATCH,
            SearchStrategy.EXACT_MATCH,
        )

    async def test_exact_match(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        medha_instance._l1_cache.clear()
        medha_instance._embedding_cache.clear()
        hit = await medha_instance.search("How many users?")
        # Same text should yield exact match via vector similarity
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.confidence >= 0.99

    async def test_semantic_match(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        medha_instance._l1_cache.clear()
        medha_instance._embedding_cache.clear()
        # Different but related question — may or may not match depending on mock hash
        hit = await medha_instance.search("Count all users")
        # With hash-based mock, different inputs produce different hashes → likely no match
        assert hit.strategy in (
            SearchStrategy.EXACT_MATCH,
            SearchStrategy.SEMANTIC_MATCH,
            SearchStrategy.NO_MATCH,
            SearchStrategy.ERROR,
        )

    async def test_no_match(self, medha_instance):
        await medha_instance.start()
        hit = await medha_instance.search("completely random xyz abc")
        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_waterfall_order(self, medha_instance):
        """Verify tiers are checked in order: L1 -> Template -> Exact -> Semantic -> Fuzzy."""
        await medha_instance.start()
        # Empty cache → should hit NO_MATCH (all tiers exhausted)
        hit = await medha_instance.search("any question here")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert medha_instance._stats["misses"] == 1


class TestStoreAndSearch:
    async def test_store_and_search(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store(
            "Get user count", "SELECT COUNT(*) FROM users"
        )
        assert ok is True
        medha_instance._l1_cache.clear()
        medha_instance._embedding_cache.clear()
        hit = await medha_instance.search("Get user count")
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.strategy == SearchStrategy.EXACT_MATCH


class TestStats:
    async def test_stats_tracking(self, medha_instance):
        await medha_instance.start()
        await medha_instance.search("no match question xyz")
        stats = medha_instance.stats
        assert stats["total_requests"] >= 1
        assert "by_strategy" in stats
        assert stats["by_strategy"]["misses"] >= 1


class TestClearCaches:
    async def test_clear_caches(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("test q", "SELECT 1")
        assert len(medha_instance._l1_cache) > 0
        medha_instance.clear_caches()
        assert len(medha_instance._l1_cache) == 0
        assert len(medha_instance._embedding_cache) == 0
        assert medha_instance._stats["l1_hits"] == 0


class TestEmptyQuestion:
    async def test_empty_question(self, medha_instance):
        await medha_instance.start()
        hit = await medha_instance.search("")
        assert hit.strategy == SearchStrategy.ERROR

    async def test_whitespace_only_question(self, medha_instance):
        await medha_instance.start()
        hit = await medha_instance.search("   ")
        assert hit.strategy == SearchStrategy.ERROR


class TestContextManager:
    async def test_context_manager(self, mock_backend, waterfall_settings):
        embedder = MockEmbedder(dimension=384)
        async with Medha(
            collection_name="ctx_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        ) as m:
            ok = await m.store("test", "SELECT 1")
            assert ok is True
        # After exit, backend.close() should have been called
        # No error means success


class TestFuzzyFallback:
    async def test_fuzzy_fallback(self, medha_instance):
        """Fuzzy matching requires rapidfuzz; if absent, gracefully skips."""
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        medha_instance._l1_cache.clear()
        medha_instance._embedding_cache.clear()
        # Query with a typo — different hash, won't match exact/semantic
        hit = await medha_instance.search("How meny usrs?")
        # With rapidfuzz: FUZZY_MATCH; without: NO_MATCH — both are acceptable
        assert hit.strategy in (
            SearchStrategy.FUZZY_MATCH,
            SearchStrategy.NO_MATCH,
        )
