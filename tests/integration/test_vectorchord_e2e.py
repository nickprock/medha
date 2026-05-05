"""End-to-end integration tests: MockEmbedder + VectorChordBackend + Medha pipeline.

Requires PostgreSQL with the vectorchord extension installed.
Skipped automatically when MEDHA_TEST_VC_DSN is not set.

Run with:
    MEDHA_TEST_VC_DSN=postgresql://postgres:postgres@localhost:5432/medha_test \
        pytest tests/integration/test_vectorchord_e2e.py -v -m vectorchord
"""

import os
import uuid

import pytest

asyncpg = pytest.importorskip("asyncpg")

from medha.backends.vectorchord import VectorChordBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy

VC_DSN = os.environ.get("MEDHA_TEST_VC_DSN")

pytestmark = pytest.mark.skipif(
    not VC_DSN,
    reason="MEDHA_TEST_VC_DSN not set — skipping VectorChord integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def medha_vc(mock_embedder):
    collection = f"test_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        backend_type="vectorchord",
        pg_dsn=VC_DSN,
        pg_table_prefix="medha_test",
        pg_schema="public",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        l1_cache_max_size=100,
    )
    backend = VectorChordBackend(settings)
    await backend.connect()

    m = Medha(
        collection_name=collection,
        embedder=mock_embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m

    # Teardown: drop test tables
    table = backend._table_name(collection)
    async with backend._pool.acquire() as conn:
        await conn.execute(f'DROP TABLE IF EXISTS public."{table}"')
    await m.close()


# ---------------------------------------------------------------------------
# Store + Search
# ---------------------------------------------------------------------------


@pytest.mark.vectorchord
class TestStoreAndSearch:
    async def test_store_and_exact_search(self, medha_vc):
        await medha_vc.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_vc.clear_caches()

        hit = await medha_vc.search("How many users are there")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.confidence >= 0.99

    async def test_store_and_semantic_search(self, medha_vc):
        await medha_vc.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_vc.clear_caches()

        hit = await medha_vc.search("How many users are there")
        assert hit.strategy in (SearchStrategy.EXACT_MATCH, SearchStrategy.SEMANTIC_MATCH)
        assert hit.generated_query == "SELECT COUNT(*) FROM users"

    async def test_l1_cache_hit(self, medha_vc):
        await medha_vc.store("Get user count", "SELECT COUNT(*) FROM users")
        await medha_vc.clear_caches()

        hit1 = await medha_vc.search("Get user count")
        assert hit1.strategy != SearchStrategy.NO_MATCH

        hit2 = await medha_vc.search("Get user count")
        assert hit2.strategy == SearchStrategy.L1_CACHE

    async def test_no_match_empty(self, medha_vc):
        hit = await medha_vc.search("What is the meaning of life")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert not hit.generated_query

    async def test_store_batch(self, medha_vc):
        collection = medha_vc._collection_name
        entries = [
            {"question": f"Question number {i}", "generated_query": f"SELECT {i}"}
            for i in range(5)
        ]
        ok = await medha_vc.store_batch(entries)
        assert ok is True

        results, _ = await medha_vc._backend.scroll(collection, limit=10)
        stored_queries = {r.generated_query for r in results}
        for i in range(5):
            assert f"SELECT {i}" in stored_queries


# ---------------------------------------------------------------------------
# Scroll
# ---------------------------------------------------------------------------


@pytest.mark.vectorchord
class TestScroll:
    async def test_scroll_all_entries(self, medha_vc):
        collection = medha_vc._collection_name
        backend = medha_vc._backend

        for i in range(5):
            await medha_vc.store(f"question {i}", f"SELECT {i}")

        results, next_offset = await backend.scroll(collection, limit=100)
        assert len(results) == 5
        assert next_offset is None

    async def test_scroll_pagination(self, medha_vc):
        collection = medha_vc._collection_name
        backend = medha_vc._backend

        for i in range(6):
            await medha_vc.store(f"question {i}", f"SELECT {i}")

        page1, next1 = await backend.scroll(collection, limit=4)
        assert len(page1) == 4
        assert next1 == "4"

        page2, next2 = await backend.scroll(collection, limit=4, offset=next1)
        assert len(page2) == 2
        assert next2 is None


# ---------------------------------------------------------------------------
# Backend instantiation via Settings
# ---------------------------------------------------------------------------


@pytest.mark.vectorchord
class TestBackendTypeViaSettings:
    async def test_backend_type_vectorchord_via_settings(self, mock_embedder):
        collection = f"auto_{uuid.uuid4().hex[:8]}"
        settings = Settings(
            backend_type="vectorchord",
            pg_dsn=VC_DSN,
            pg_table_prefix="medha_test",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
        )
        m = Medha(
            collection_name=collection,
            embedder=mock_embedder,
            settings=settings,
        )
        assert isinstance(m._backend, VectorChordBackend)
        await m.start()

        # Teardown
        table = m._backend._table_name(collection)
        async with m._backend._pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS public."{table}"')
        await m.close()
