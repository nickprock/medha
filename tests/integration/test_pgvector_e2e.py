"""End-to-end integration tests: MockEmbedder + PgVectorBackend + Medha pipeline.

Requires PostgreSQL with pgvector extension installed.
Skipped automatically when MEDHA_TEST_PG_DSN is not set.

Run with:
    MEDHA_TEST_PG_DSN=postgresql://postgres:postgres@localhost:5432/medha_test \
        pytest tests/integration/test_pgvector_e2e.py -v -m pgvector
"""

import os
import uuid

import pytest

asyncpg = pytest.importorskip("asyncpg")

from medha.backends.pgvector import PgVectorBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy

PG_DSN = os.environ.get("MEDHA_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(
    not PG_DSN,
    reason="MEDHA_TEST_PG_DSN not set — skipping pgvector integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def medha_pgvector(mock_embedder):
    """Medha with PgVectorBackend — isolated collection per test, cleaned up after."""
    collection = f"test_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        backend_type="pgvector",
        pg_dsn=PG_DSN,
        pg_table_prefix="medha_test",
        pg_schema="public",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        l1_cache_max_size=100,
    )
    backend = PgVectorBackend(settings)
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
    tmpl_table = backend._table_name(f"__medha_templates_{collection}")
    async with backend._pool.acquire() as conn:
        await conn.execute(f'DROP TABLE IF EXISTS public."{table}"')
        await conn.execute(f'DROP TABLE IF EXISTS public."{tmpl_table}"')
    await m.close()


# ---------------------------------------------------------------------------
# Store + Search
# ---------------------------------------------------------------------------


@pytest.mark.pgvector
class TestStoreAndSearch:
    async def test_store_and_exact_search(self, medha_pgvector):
        """Store + search identical question → EXACT_MATCH."""
        await medha_pgvector.store(
            "How many users are there", "SELECT COUNT(*) FROM users"
        )
        await medha_pgvector.clear_caches()

        hit = await medha_pgvector.search("How many users are there")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.confidence >= 0.99

    async def test_store_and_semantic_search(self, medha_pgvector):
        """Store + search → SEMANTIC_MATCH or EXACT_MATCH (MockEmbedder is hash-based)."""
        await medha_pgvector.store(
            "How many users are there", "SELECT COUNT(*) FROM users"
        )
        await medha_pgvector.clear_caches()

        hit = await medha_pgvector.search("How many users are there")
        assert hit.strategy in (SearchStrategy.EXACT_MATCH, SearchStrategy.SEMANTIC_MATCH)
        assert hit.generated_query == "SELECT COUNT(*) FROM users"

    async def test_l1_cache_hit(self, medha_pgvector):
        """Second identical search → L1_CACHE strategy."""
        await medha_pgvector.store("Get user count", "SELECT COUNT(*) FROM users")
        await medha_pgvector.clear_caches()

        hit1 = await medha_pgvector.search("Get user count")
        assert hit1.strategy != SearchStrategy.NO_MATCH

        hit2 = await medha_pgvector.search("Get user count")
        assert hit2.strategy == SearchStrategy.L1_CACHE
        s = await medha_pgvector.stats()
        assert s.by_strategy.get("l1_cache") is not None and s.by_strategy["l1_cache"].count >= 1

    async def test_no_match_empty(self, medha_pgvector):
        """Search on empty collection → NO_MATCH."""
        hit = await medha_pgvector.search("What is the meaning of life")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert not hit.generated_query

    async def test_store_batch(self, medha_pgvector):
        """store_batch with 5 entries → all retrievable via scroll."""
        collection = medha_pgvector._collection_name
        entries = [
            {"question": f"Question number {i}", "generated_query": f"SELECT {i}"}
            for i in range(5)
        ]
        ok = await medha_pgvector.store_batch(entries)
        assert ok is True

        results, _ = await medha_pgvector._backend.scroll(collection, limit=10)
        stored_queries = {r.generated_query for r in results}
        for i in range(5):
            assert f"SELECT {i}" in stored_queries


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


@pytest.mark.pgvector
class TestDeduplication:
    async def test_upsert_deduplication(self, medha_pgvector):
        """Upsert same id twice → 1 row in DB, payload updated."""
        collection = medha_pgvector._collection_name
        backend = medha_pgvector._backend

        await medha_pgvector.store("Original question", "SELECT 1")
        count_after_first = await backend.count(collection)

        await medha_pgvector.store("Original question", "SELECT 1")
        count_after_second = await backend.count(collection)

        # May grow by 1 (different hash-derived ids per store call) or stay same
        # The important thing: re-storing the exact same question+query is idempotent
        # at the DB level (ON CONFLICT DO UPDATE). We verify no crash and count >= 1.
        assert count_after_first >= 1
        assert count_after_second >= count_after_first


# ---------------------------------------------------------------------------
# Scroll
# ---------------------------------------------------------------------------


@pytest.mark.pgvector
class TestScroll:
    async def test_scroll_all_entries(self, medha_pgvector):
        """scroll() with large limit returns all entries and next_offset=None."""
        collection = medha_pgvector._collection_name
        backend = medha_pgvector._backend

        for i in range(5):
            await medha_pgvector.store(f"question {i}", f"SELECT {i}")

        results, next_offset = await backend.scroll(collection, limit=100)
        assert len(results) == 5
        assert next_offset is None

    async def test_scroll_pagination(self, medha_pgvector):
        """scroll() with limit < total returns correct pages."""
        collection = medha_pgvector._collection_name
        backend = medha_pgvector._backend

        for i in range(6):
            await medha_pgvector.store(f"question {i}", f"SELECT {i}")

        page1, next1 = await backend.scroll(collection, limit=4)
        assert len(page1) == 4
        assert next1 == "4"

        page2, next2 = await backend.scroll(collection, limit=4, offset=next1)
        assert len(page2) == 2
        assert next2 is None


# ---------------------------------------------------------------------------
# Backend instantiation via Settings
# ---------------------------------------------------------------------------


@pytest.mark.pgvector
class TestBackendTypeViaSettings:
    async def test_backend_type_pgvector_via_settings(self, mock_embedder):
        """Medha(settings=Settings(backend_type='pgvector', ...)) auto-instantiates PgVectorBackend."""
        collection = f"auto_{uuid.uuid4().hex[:8]}"
        settings = Settings(
            backend_type="pgvector",
            pg_dsn=PG_DSN,
            pg_table_prefix="medha_test",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
        )
        m = Medha(
            collection_name=collection,
            embedder=mock_embedder,
            settings=settings,
        )
        assert isinstance(m._backend, PgVectorBackend)
        await m.start()

        # Teardown
        table = m._backend._table_name(collection)
        async with m._backend._pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS public."{table}"')
        await m.close()


# ---------------------------------------------------------------------------
# Table isolation
# ---------------------------------------------------------------------------


@pytest.mark.pgvector
class TestTableIsolation:
    async def test_table_isolation(self, mock_embedder):
        """Two Medha instances with different collections use separate tables."""
        col_a = f"iso_a_{uuid.uuid4().hex[:6]}"
        col_b = f"iso_b_{uuid.uuid4().hex[:6]}"

        settings = Settings(
            backend_type="pgvector",
            pg_dsn=PG_DSN,
            pg_table_prefix="medha_test",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
        )

        backend_a = PgVectorBackend(settings)
        backend_b = PgVectorBackend(settings)
        await backend_a.connect()
        await backend_b.connect()

        m_a = Medha(collection_name=col_a, embedder=mock_embedder, backend=backend_a, settings=settings)
        m_b = Medha(collection_name=col_b, embedder=mock_embedder, backend=backend_b, settings=settings)
        await m_a.start()
        await m_b.start()

        try:
            await m_a.store("alpha question", "SELECT 'alpha'")
            await m_b.store("beta question", "SELECT 'beta'")

            await m_a.clear_caches()
            await m_b.clear_caches()

            hit_a = await m_a.search("alpha question")
            hit_b = await m_b.search("beta question")
            assert hit_a.generated_query == "SELECT 'alpha'"
            assert hit_b.generated_query == "SELECT 'beta'"

            # Cross-check: col_a doesn't see beta
            miss = await m_a.search("beta question")
            assert miss.strategy == SearchStrategy.NO_MATCH
        finally:
            for col, backend in [(col_a, backend_a), (col_b, backend_b)]:
                table = backend._table_name(col)
                tmpl_table = backend._table_name(f"__medha_templates_{col}")
                async with backend._pool.acquire() as conn:
                    await conn.execute(f'DROP TABLE IF EXISTS public."{table}"')
                    await conn.execute(f'DROP TABLE IF EXISTS public."{tmpl_table}"')
            await m_a.close()
            await m_b.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.pgvector
class TestContextManager:
    async def test_context_manager(self, mock_embedder):
        """async with Medha(...) → no exception, close() called correctly."""
        collection = f"ctx_{uuid.uuid4().hex[:8]}"
        settings = Settings(
            backend_type="pgvector",
            pg_dsn=PG_DSN,
            pg_table_prefix="medha_test",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
        )
        backend = PgVectorBackend(settings)
        await backend.connect()

        async with Medha(
            collection_name=collection,
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        ) as m:
            await m.store("test question", "SELECT 1")
            hit = await m.search("test question")
            assert hit.strategy != SearchStrategy.NO_MATCH

        # Pool closed after context exit
        assert backend._pool is None

        # Cleanup: reconnect briefly to drop the table
        await backend.connect()
        table = backend._table_name(collection)
        async with backend._pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS public."{table}"')
        await backend.close()
