"""Unit tests for InMemoryBackend."""

import asyncio
import hashlib
import uuid

import pytest

from medha.backends.memory import InMemoryBackend, _cosine_similarity
from medha.exceptions import StorageError
from medha.types import CacheEntry, CacheResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    id: str | None = None,
    vector: list[float] | None = None,
    question: str = "test question",
    query: str = "SELECT 1",
    dim: int = 8,
) -> CacheEntry:
    vec = vector or [0.1] * dim
    return CacheEntry(
        id=id or str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=hashlib.md5(query.encode()).hexdigest(),
    )


COLL = "test_collection"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    async def test_connect_is_noop(self):
        b = InMemoryBackend()
        await b.connect()  # must not raise

    async def test_close_clears_store(self):
        b = InMemoryBackend()
        await b.initialize(COLL, 8)
        await b.upsert(COLL, [_make_entry()])
        await b.close()
        # After close, count returns 0 (collection gone)
        assert await b.count(COLL) == 0

    async def test_context_manager(self):
        async with InMemoryBackend() as b:
            await b.initialize(COLL, 8)
            await b.upsert(COLL, [_make_entry()])
            assert await b.count(COLL) == 1
        # close() was called; store cleared
        assert await b.count(COLL) == 0


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

class TestInitialize:
    async def test_initialize_creates_collection(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        assert await inmemory_backend.count(COLL) == 0

    async def test_initialize_idempotent(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        await inmemory_backend.upsert(COLL, [_make_entry()])
        # Second initialize must not reset the collection
        await inmemory_backend.initialize(COLL, 8)
        assert await inmemory_backend.count(COLL) == 1

    async def test_initialize_multiple_collections(self, inmemory_backend):
        await inmemory_backend.initialize("col_a", 8)
        await inmemory_backend.initialize("col_b", 8)
        await inmemory_backend.upsert("col_a", [_make_entry()])
        assert await inmemory_backend.count("col_a") == 1
        assert await inmemory_backend.count("col_b") == 0

    async def test_initialize_invalid_dimension(self, inmemory_backend):
        with pytest.raises(StorageError):
            await inmemory_backend.initialize(COLL, 0)


# ---------------------------------------------------------------------------
# upsert + count
# ---------------------------------------------------------------------------

class TestUpsertCount:
    async def test_upsert_single_entry(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        await inmemory_backend.upsert(COLL, [_make_entry()])
        assert await inmemory_backend.count(COLL) == 1

    async def test_upsert_multiple_entries(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        entries = [_make_entry() for _ in range(5)]
        await inmemory_backend.upsert(COLL, entries)
        assert await inmemory_backend.count(COLL) == 5

    async def test_upsert_same_id_overwrites(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        eid = str(uuid.uuid4())
        await inmemory_backend.upsert(COLL, [_make_entry(id=eid, query="SELECT 1")])
        await inmemory_backend.upsert(COLL, [_make_entry(id=eid, query="SELECT 2")])
        assert await inmemory_backend.count(COLL) == 1
        # Check payload updated
        results, _ = await inmemory_backend.scroll(COLL)
        assert results[0].generated_query == "SELECT 2"

    async def test_upsert_empty_list(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        await inmemory_backend.upsert(COLL, [])  # must not raise
        assert await inmemory_backend.count(COLL) == 0

    async def test_upsert_uninitialized_collection(self, inmemory_backend):
        with pytest.raises(StorageError):
            await inmemory_backend.upsert("nonexistent", [_make_entry()])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    async def test_search_returns_sorted_by_score(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 4)
        # query vector ~ [1, 0, 0, 0]
        query = [1.0, 0.0, 0.0, 0.0]
        # close match
        await inmemory_backend.upsert(COLL, [_make_entry(id="a", vector=[0.9, 0.1, 0.0, 0.0], query="SELECT A")])
        # far match
        await inmemory_backend.upsert(COLL, [_make_entry(id="b", vector=[0.0, 0.0, 0.0, 1.0], query="SELECT B")])
        results = await inmemory_backend.search(COLL, query, limit=5)
        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].generated_query == "SELECT A"

    async def test_search_respects_score_threshold(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 4)
        query = [1.0, 0.0, 0.0, 0.0]
        await inmemory_backend.upsert(COLL, [_make_entry(id="a", vector=[1.0, 0.0, 0.0, 0.0])])
        await inmemory_backend.upsert(COLL, [_make_entry(id="b", vector=[0.0, 1.0, 0.0, 0.0])])
        # orthogonal vector has ~0 score, should be filtered
        results = await inmemory_backend.search(COLL, query, score_threshold=0.5)
        assert all(r.score >= 0.5 for r in results)
        assert not any(r.id == "b" for r in results)

    async def test_search_respects_limit(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        entries = [_make_entry() for _ in range(10)]
        await inmemory_backend.upsert(COLL, entries)
        results = await inmemory_backend.search(COLL, [0.1] * 8, limit=3)
        assert len(results) <= 3

    async def test_search_empty_collection(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        results = await inmemory_backend.search(COLL, [0.1] * 8)
        assert results == []

    async def test_search_identical_vector(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 4)
        vec = [1.0, 0.0, 0.0, 0.0]
        await inmemory_backend.upsert(COLL, [_make_entry(vector=vec)])
        results = await inmemory_backend.search(COLL, vec, limit=1)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_search_orthogonal_vectors(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 4)
        await inmemory_backend.upsert(COLL, [_make_entry(vector=[0.0, 1.0, 0.0, 0.0])])
        results = await inmemory_backend.search(COLL, [1.0, 0.0, 0.0, 0.0], score_threshold=0.0)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.0, abs=1e-5)

    async def test_search_zero_vector_no_crash(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 4)
        await inmemory_backend.upsert(COLL, [_make_entry(vector=[0.0, 0.0, 0.0, 0.0])])
        results = await inmemory_backend.search(COLL, [0.0, 0.0, 0.0, 0.0], score_threshold=0.0)
        assert len(results) == 1
        assert results[0].score == 0.0

    async def test_search_uninitialized_collection(self, inmemory_backend):
        with pytest.raises(StorageError):
            await inmemory_backend.search("nonexistent", [0.1] * 8)


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------

class TestScroll:
    async def test_scroll_all_entries_single_page(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        entries = [_make_entry() for _ in range(5)]
        await inmemory_backend.upsert(COLL, entries)
        results, next_offset = await inmemory_backend.scroll(COLL, limit=10)
        assert len(results) == 5
        assert next_offset is None

    async def test_scroll_pagination(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        entries = [_make_entry() for _ in range(7)]
        await inmemory_backend.upsert(COLL, entries)
        page1, next_offset = await inmemory_backend.scroll(COLL, limit=4)
        assert len(page1) == 4
        assert next_offset == "4"
        page2, next_offset2 = await inmemory_backend.scroll(COLL, limit=4, offset=next_offset)
        assert len(page2) == 3
        assert next_offset2 is None

    async def test_scroll_last_page(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        entries = [_make_entry() for _ in range(3)]
        await inmemory_backend.upsert(COLL, entries)
        _, next_offset = await inmemory_backend.scroll(COLL, limit=10, offset="0")
        assert next_offset is None

    async def test_scroll_empty_collection(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        results, next_offset = await inmemory_backend.scroll(COLL, limit=10)
        assert results == []
        assert next_offset is None


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    async def test_delete_existing_ids(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        e1, e2 = _make_entry(), _make_entry()
        await inmemory_backend.upsert(COLL, [e1, e2])
        await inmemory_backend.delete(COLL, [e1.id])
        assert await inmemory_backend.count(COLL) == 1

    async def test_delete_unknown_id_is_noop(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        await inmemory_backend.delete(COLL, ["nonexistent-id"])  # must not raise

    async def test_delete_empty_list(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        await inmemory_backend.delete(COLL, [])  # must not raise


# ---------------------------------------------------------------------------
# Extra methods
# ---------------------------------------------------------------------------

class TestExtraMethods:
    async def test_search_by_query_hash_found(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        entry = _make_entry(query="SELECT 42")
        await inmemory_backend.upsert(COLL, [entry])
        result = await inmemory_backend.search_by_query_hash(COLL, entry.query_hash)
        assert result is not None
        assert isinstance(result, CacheResult)
        assert result.score == 1.0
        assert result.query_hash == entry.query_hash

    async def test_search_by_query_hash_not_found(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        result = await inmemory_backend.search_by_query_hash(COLL, "deadbeef")
        assert result is None

    async def test_update_usage_count_increments(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        entry = _make_entry()
        await inmemory_backend.upsert(COLL, [entry])
        await inmemory_backend.update_usage_count(COLL, entry.id)
        results, _ = await inmemory_backend.scroll(COLL)
        assert results[0].usage_count == 2

    async def test_update_usage_count_unknown_id(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        await inmemory_backend.update_usage_count(COLL, "nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# Cosine similarity unit tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0, 4.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    async def test_concurrent_upsert(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)

        async def _upsert_one(i: int):
            entry = _make_entry(id=f"entry-{i}")
            await inmemory_backend.upsert(COLL, [entry])

        await asyncio.gather(*[_upsert_one(i) for i in range(10)])
        assert await inmemory_backend.count(COLL) == 10

    async def test_concurrent_search_and_upsert(self, inmemory_backend):
        await inmemory_backend.initialize(COLL, 8)
        await inmemory_backend.upsert(COLL, [_make_entry()])

        async def _search():
            for _ in range(5):
                await inmemory_backend.search(COLL, [0.1] * 8)

        async def _upsert():
            for i in range(5):
                await inmemory_backend.upsert(COLL, [_make_entry(id=f"concurrent-{i}")])

        await asyncio.gather(_search(), _upsert())  # must not crash
