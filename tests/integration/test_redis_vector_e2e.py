"""End-to-end integration tests: MockEmbedder + RedisVectorBackend + Medha pipeline.

Requires Redis Stack (with RediSearch module) running.
Skipped automatically when MEDHA_TEST_REDIS_URL is not set.

Run with:
    MEDHA_TEST_REDIS_URL=redis://localhost:6379/0 \
        pytest tests/integration/test_redis_vector_e2e.py -v -m redis
"""

import os
import uuid

import pytest

redis_lib = pytest.importorskip("redis")
numpy_lib = pytest.importorskip("numpy")

from medha.backends.redis_vector import RedisVectorBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy

REDIS_URL = os.environ.get("MEDHA_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(
    not REDIS_URL,
    reason="MEDHA_TEST_REDIS_URL not set — skipping Redis integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def redis_backend():
    settings = Settings(
        backend_type="redis",
        redis_url=REDIS_URL,
    )
    b = RedisVectorBackend(settings)
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
async def medha_redis(mock_embedder):
    collection = f"test_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        backend_type="redis",
        redis_url=REDIS_URL,
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        l1_cache_max_size=100,
    )
    backend = RedisVectorBackend(settings)
    await backend.connect()

    m = Medha(
        collection_name=collection,
        embedder=mock_embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m

    # Teardown: drop test index and keys
    await backend.drop_collection(collection)
    await m.close()


# ---------------------------------------------------------------------------
# Backend-level tests
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestInitialize:
    async def test_initialize_creates_index(self, redis_backend):
        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)
        count = await redis_backend.count(collection)
        assert count == 0
        await redis_backend.drop_collection(collection)

    async def test_initialize_idempotent(self, redis_backend):
        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)
        await redis_backend.initialize(collection, 384)
        assert await redis_backend.count(collection) == 0
        await redis_backend.drop_collection(collection)


@pytest.mark.redis
class TestUpsertAndSearch:
    async def test_upsert_and_search(self, redis_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("how many users?")
        entry = make_entry(vector=vec, question="How many users?", query="SELECT COUNT(*) FROM users", dim=384)
        await redis_backend.upsert(collection, [entry])

        results = await redis_backend.search(collection, vec, limit=5, score_threshold=0.0)
        assert len(results) >= 1
        assert results[0].generated_query == "SELECT COUNT(*) FROM users"
        assert results[0].score > 0.9

        await redis_backend.drop_collection(collection)

    async def test_search_threshold(self, redis_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("how many users?")
        entry = make_entry(vector=vec, question="How many users?", query="SELECT COUNT(*) FROM users", dim=384)
        await redis_backend.upsert(collection, [entry])

        unrelated_vec = await mock_embedder.aembed("completely unrelated xyz 12345")
        results = await redis_backend.search(collection, unrelated_vec, limit=5, score_threshold=0.99)
        assert len(results) == 0

        await redis_backend.drop_collection(collection)


@pytest.mark.redis
class TestScrollAndDelete:
    async def test_scroll_pagination(self, redis_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)

        entries = []
        for i in range(5):
            vec = await mock_embedder.aembed(f"question {i}")
            e = make_entry(vector=vec, question=f"question {i}", query=f"SELECT {i}", dim=384)
            entries.append(e)
        await redis_backend.upsert(collection, entries)

        all_results = []
        offset = None
        while True:
            batch, offset = await redis_backend.scroll(collection, limit=2, offset=offset)
            all_results.extend(batch)
            if offset is None:
                break

        assert len(all_results) == 5
        await redis_backend.drop_collection(collection)

    async def test_delete(self, redis_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("test")
        entry = make_entry(vector=vec, dim=384)
        await redis_backend.upsert(collection, [entry])

        assert await redis_backend.count(collection) == 1
        await redis_backend.delete(collection, [entry.id])
        assert await redis_backend.count(collection) == 0

        await redis_backend.drop_collection(collection)


@pytest.mark.redis
class TestSearchByQueryHash:
    async def test_search_by_query_hash(self, redis_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("test")
        entry = make_entry(vector=vec, query="SELECT 99", dim=384)
        await redis_backend.upsert(collection, [entry])

        result = await redis_backend.search_by_query_hash(collection, entry.query_hash)
        assert result is not None
        assert result.generated_query == "SELECT 99"

        await redis_backend.drop_collection(collection)

    async def test_search_by_query_hash_not_found(self, redis_backend, mock_embedder):
        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)

        result = await redis_backend.search_by_query_hash(collection, "nonexistent_hash")
        assert result is None

        await redis_backend.drop_collection(collection)


@pytest.mark.redis
class TestUpdateUsageCount:
    async def test_update_usage_count(self, redis_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await redis_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("test")
        entry = make_entry(vector=vec, dim=384)
        await redis_backend.upsert(collection, [entry])

        await redis_backend.update_usage_count(collection, entry.id)

        results = await redis_backend.search(collection, vec, limit=1)
        assert results[0].usage_count == 2

        await redis_backend.drop_collection(collection)


# ---------------------------------------------------------------------------
# Medha pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestStoreAndSearch:
    async def test_store_and_exact_search(self, medha_redis):
        await medha_redis.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_redis.clear_caches()

        hit = await medha_redis.search("How many users are there")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.generated_query == "SELECT COUNT(*) FROM users"

    async def test_no_match_empty(self, medha_redis):
        hit = await medha_redis.search("What is the meaning of life")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert not hit.generated_query

    async def test_l1_cache_hit(self, medha_redis):
        await medha_redis.store("Get user count", "SELECT COUNT(*) FROM users")
        await medha_redis.clear_caches()

        hit1 = await medha_redis.search("Get user count")
        assert hit1.strategy != SearchStrategy.NO_MATCH

        hit2 = await medha_redis.search("Get user count")
        assert hit2.strategy == SearchStrategy.L1_CACHE
