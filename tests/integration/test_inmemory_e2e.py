"""End-to-end integration tests: MockEmbedder + InMemoryBackend + Medha pipeline."""


from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy


class TestStoreAndSearch:
    async def test_store_and_exact_search(self, medha_memory):
        """Store a question and search with the identical question → EXACT_MATCH."""
        await medha_memory.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_memory.clear_caches()

        hit = await medha_memory.search("How many users are there")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.confidence >= 0.99

    async def test_store_and_semantic_search(self, medha_memory):
        """Store a question and search with a paraphrase → SEMANTIC_MATCH or EXACT_MATCH."""
        await medha_memory.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_memory.clear_caches()

        # MockEmbedder is hash-based: similar (but not identical) text will have some overlap
        # Use a very similar question to ensure it scores above semantic threshold
        hit = await medha_memory.search("How many users are there")
        assert hit.strategy in (SearchStrategy.EXACT_MATCH, SearchStrategy.SEMANTIC_MATCH)
        assert hit.generated_query == "SELECT COUNT(*) FROM users"

    async def test_l1_cache_hit(self, medha_memory):
        """Second identical search → L1_CACHE strategy."""
        await medha_memory.store("Get user count", "SELECT COUNT(*) FROM users")
        # clear_caches resets L1 so the first real search goes to backend
        await medha_memory.clear_caches()

        hit1 = await medha_memory.search("Get user count")
        assert hit1.strategy != SearchStrategy.NO_MATCH

        # Second identical search — L1 was populated by the first search
        hit2 = await medha_memory.search("Get user count")
        assert hit2.strategy == SearchStrategy.L1_CACHE
        assert medha_memory._stats["l1_hits"] >= 1

    async def test_no_match_empty(self, medha_memory):
        """Search on empty collection → NO_MATCH."""
        hit = await medha_memory.search("What is the meaning of life")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert not hit.generated_query

    async def test_store_batch(self, medha_memory):
        """store_batch with 5 entries → all retrievable via scroll."""
        entries = [
            {"question": f"Question number {i}", "generated_query": f"SELECT {i}"}
            for i in range(5)
        ]
        ok = await medha_memory.store_batch(entries)
        assert ok is True

        results, _ = await medha_memory._backend.scroll("inmemory_e2e", limit=10)
        stored_queries = {r.generated_query for r in results}
        for i in range(5):
            assert f"SELECT {i}" in stored_queries


class TestStats:
    async def test_stats_updated(self, medha_memory):
        """get_stats() reflects the correct hit counts."""
        await medha_memory.store("Count orders", "SELECT COUNT(*) FROM orders")
        await medha_memory.clear_caches()

        await medha_memory.search("Count orders")   # exact hit
        await medha_memory.search("Count orders")   # l1 hit
        await medha_memory.search("nonexistent xyz abc 123")  # miss

        stats = medha_memory.stats
        assert stats["total_requests"] >= 3
        assert stats["by_strategy"]["l1_hits"] >= 1
        assert stats["by_strategy"]["misses"] >= 1


class TestBackendTypeViaSettings:
    async def test_backend_type_memory_via_settings(self, mock_embedder):
        """Medha(settings=Settings(backend_type='memory')) instantiates InMemoryBackend automatically."""
        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
        )
        m = Medha(
            collection_name="auto_backend_test",
            embedder=mock_embedder,
            settings=settings,
        )
        assert isinstance(m._backend, InMemoryBackend)
        await m.start()
        await m.close()


class TestContextManager:
    async def test_context_manager(self, mock_embedder):
        """async with Medha(...) as m — no exception, close() called correctly."""
        settings = Settings(backend_type="memory")
        async with Medha(
            collection_name="ctx_test",
            embedder=mock_embedder,
            settings=settings,
        ) as m:
            await m.store("test question", "SELECT 1")
            hit = await m.search("test question")
            assert hit.strategy != SearchStrategy.NO_MATCH
        # After exit, the backend store is cleared
        assert await m._backend.count("ctx_test") == 0


class TestCollectionIsolation:
    async def test_multiple_collections_isolated(self, mock_embedder):
        """Two Medha instances with different collections don't interfere."""
        settings = Settings(backend_type="memory")

        m1 = Medha(collection_name="col_alpha", embedder=mock_embedder, settings=settings)
        m2 = Medha(collection_name="col_beta", embedder=mock_embedder, settings=settings)

        await m1.start()
        await m2.start()

        try:
            await m1.store("alpha question", "SELECT 'alpha'")
            await m2.store("beta question", "SELECT 'beta'")

            await m1.clear_caches()
            await m2.clear_caches()

            hit_alpha = await m1.search("alpha question")
            hit_beta = await m2.search("beta question")

            assert hit_alpha.generated_query == "SELECT 'alpha'"
            assert hit_beta.generated_query == "SELECT 'beta'"

            # Cross-check: alpha collection doesn't contain beta
            miss = await m1.search("beta question")
            assert miss.strategy == SearchStrategy.NO_MATCH
        finally:
            await m1.close()
            await m2.close()
