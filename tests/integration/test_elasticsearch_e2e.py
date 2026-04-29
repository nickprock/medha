"""End-to-end integration tests: MockEmbedder + ElasticsearchBackend + Medha pipeline.

Requires a running Elasticsearch 8.x instance.
Skipped automatically when MEDHA_TEST_ES_HOSTS is not set.

Run with:
    MEDHA_TEST_ES_HOSTS=http://localhost:9200 \
        pytest tests/integration/test_elasticsearch_e2e.py -v -m elasticsearch
"""

import os
import uuid

import pytest

elasticsearch = pytest.importorskip("elasticsearch")

from medha.backends.elasticsearch import ElasticsearchBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy

ES_HOSTS_RAW = os.environ.get("MEDHA_TEST_ES_HOSTS")
ES_HOSTS = [h.strip() for h in ES_HOSTS_RAW.split(",")] if ES_HOSTS_RAW else None

pytestmark = pytest.mark.skipif(
    not ES_HOSTS,
    reason="MEDHA_TEST_ES_HOSTS not set — skipping Elasticsearch integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def es_backend():
    settings = Settings(
        backend_type="elasticsearch",
        es_hosts=ES_HOSTS or ["http://localhost:9200"],
    )
    b = ElasticsearchBackend(settings)
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
async def medha_es(mock_embedder):
    collection = f"test_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        backend_type="elasticsearch",
        es_hosts=ES_HOSTS or ["http://localhost:9200"],
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        l1_cache_max_size=100,
    )
    backend = ElasticsearchBackend(settings)
    await backend.connect()

    m = Medha(
        collection_name=collection,
        embedder=mock_embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m

    # Teardown: drop test index
    await backend.drop_collection(collection)
    await m.close()


# ---------------------------------------------------------------------------
# Backend-level tests
# ---------------------------------------------------------------------------


@pytest.mark.elasticsearch
class TestInitialize:
    async def test_initialize_creates_index(self, es_backend):
        collection = f"test_{uuid.uuid4().hex[:8]}"
        await es_backend.initialize(collection, 384)
        count = await es_backend.count(collection)
        assert count == 0
        await es_backend.drop_collection(collection)

    async def test_initialize_idempotent(self, es_backend):
        collection = f"test_{uuid.uuid4().hex[:8]}"
        await es_backend.initialize(collection, 384)
        await es_backend.initialize(collection, 384)
        assert await es_backend.count(collection) == 0
        await es_backend.drop_collection(collection)


@pytest.mark.elasticsearch
class TestUpsertAndSearch:
    async def test_upsert_and_search(self, es_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await es_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("how many users?")
        entry = make_entry(vector=vec, question="How many users?", query="SELECT COUNT(*) FROM users", dim=384)
        await es_backend.upsert(collection, [entry])

        # Elasticsearch needs a refresh before searching
        await es_backend._client.indices.refresh(index=es_backend._index_name(collection))

        results = await es_backend.search(collection, vec, limit=5, score_threshold=0.0)
        assert len(results) >= 1
        assert results[0].generated_query == "SELECT COUNT(*) FROM users"

        await es_backend.drop_collection(collection)

    async def test_search_threshold(self, es_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await es_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("how many users?")
        entry = make_entry(vector=vec, question="How many users?", query="SELECT COUNT(*) FROM users", dim=384)
        await es_backend.upsert(collection, [entry])

        await es_backend._client.indices.refresh(index=es_backend._index_name(collection))

        unrelated_vec = await mock_embedder.aembed("completely unrelated xyz 12345")
        results = await es_backend.search(collection, unrelated_vec, limit=5, score_threshold=0.99)
        assert len(results) == 0

        await es_backend.drop_collection(collection)


@pytest.mark.elasticsearch
class TestScrollAndDelete:
    async def test_scroll_pagination(self, es_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await es_backend.initialize(collection, 384)

        entries = []
        for i in range(5):
            vec = await mock_embedder.aembed(f"question {i}")
            e = make_entry(vector=vec, question=f"question {i}", query=f"SELECT {i}", dim=384)
            entries.append(e)
        await es_backend.upsert(collection, entries)
        await es_backend._client.indices.refresh(index=es_backend._index_name(collection))

        all_results = []
        offset = None
        while True:
            batch, offset = await es_backend.scroll(collection, limit=2, offset=offset)
            all_results.extend(batch)
            if offset is None:
                break

        assert len(all_results) == 5
        await es_backend.drop_collection(collection)

    async def test_delete(self, es_backend, mock_embedder):
        from tests.conftest import make_entry

        collection = f"test_{uuid.uuid4().hex[:8]}"
        await es_backend.initialize(collection, 384)

        vec = await mock_embedder.aembed("test")
        entry = make_entry(vector=vec, dim=384)
        await es_backend.upsert(collection, [entry])
        await es_backend._client.indices.refresh(index=es_backend._index_name(collection))

        assert await es_backend.count(collection) == 1
        await es_backend.delete(collection, [entry.id])
        await es_backend._client.indices.refresh(index=es_backend._index_name(collection))
        assert await es_backend.count(collection) == 0

        await es_backend.drop_collection(collection)


# ---------------------------------------------------------------------------
# Medha pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.elasticsearch
class TestStoreAndSearch:
    async def test_store_and_exact_search(self, medha_es):
        await medha_es.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_es.clear_caches()

        # Force index refresh so search can find the document
        await medha_es._backend._client.indices.refresh(
            index=medha_es._backend._index_name(medha_es._collection_name)
        )

        hit = await medha_es.search("How many users are there")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.generated_query == "SELECT COUNT(*) FROM users"

    async def test_no_match_empty(self, medha_es):
        hit = await medha_es.search("What is the meaning of life")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert not hit.generated_query
