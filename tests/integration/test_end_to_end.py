"""End-to-end integration tests: FastEmbed + Qdrant memory + Medha pipeline."""

import pytest

fastembed = pytest.importorskip("fastembed")

from medha.backends.qdrant import QdrantBackend
from medha.config import Settings
from medha.core import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter
from medha.types import SearchStrategy


@pytest.fixture
def settings():
    return Settings(
        qdrant_mode="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


@pytest.fixture
def embedder():
    return FastEmbedAdapter()


@pytest.fixture
async def medha(embedder, settings):
    backend = QdrantBackend(settings)
    await backend.connect()
    m = Medha(
        collection_name="e2e_test",
        embedder=embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


class TestStoreSQLAndRetrieve:
    async def test_store_sql_and_retrieve(self, medha):
        ok = await medha.store(
            "How many users are there",
            "SELECT COUNT(*) FROM users",
        )
        assert ok is True

        medha._l1_cache.clear()
        medha._embedding_cache.clear()

        # "Number of users" has ~0.90 cosine sim with "How many users are there"
        hit = await medha.search("Number of users")
        assert hit.strategy in (
            SearchStrategy.EXACT_MATCH,
            SearchStrategy.SEMANTIC_MATCH,
        )
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.confidence > 0.0


class TestStoreCypherAndRetrieve:
    async def test_store_cypher_and_retrieve(self, medha):
        ok = await medha.store(
            "Find John's friends",
            "MATCH (p:Person {name:'John'})-[:FRIEND]-(f) RETURN f",
        )
        assert ok is True

        medha._l1_cache.clear()
        medha._embedding_cache.clear()

        hit = await medha.search("Find John's friends")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert "MATCH" in hit.generated_query


class TestMultipleStoresAndRanking:
    async def test_multiple_stores_and_ranking(self, medha):
        await medha.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha.store("List all products", "SELECT * FROM products")
        await medha.store("Show employee names", "SELECT name FROM employees")

        medha._l1_cache.clear()
        medha._embedding_cache.clear()

        # "How many users exist" has ~0.94 cosine sim with "How many users are there"
        hit = await medha.search("How many users exist")
        assert hit.strategy in (
            SearchStrategy.EXACT_MATCH,
            SearchStrategy.SEMANTIC_MATCH,
        )
        assert "users" in hit.generated_query.lower()


class TestL1CacheOnRepeat:
    async def test_l1_cache_on_repeat(self, medha):
        await medha.store("Get user count", "SELECT COUNT(*) FROM users")

        # First search — hits backend (but also goes through L1 which was set by store)
        hit1 = await medha.search("Get user count")
        assert hit1.strategy != SearchStrategy.NO_MATCH

        # Second search — should be L1 hit
        hit2 = await medha.search("Get user count")
        assert hit2.strategy != SearchStrategy.NO_MATCH
        assert medha._stats["l1_hits"] >= 1
