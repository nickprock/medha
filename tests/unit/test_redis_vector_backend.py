"""Unit tests for RedisVectorBackend (mocked redis async client — no real Redis required)."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

redis = pytest.importorskip("redis")
numpy = pytest.importorskip("numpy")

from medha.config import Settings
from medha.exceptions import ConfigurationError, StorageError
from medha.types import CacheEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLL = "test_collection"
DIM = 8


def _make_entry(
    id: str | None = None,
    vector: list[float] | None = None,
    question: str = "test question",
    query: str = "SELECT 1",
    dim: int = DIM,
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


def _redis_settings(**overrides) -> Settings:
    return Settings(backend_type="redis", **overrides)


def _make_doc(id_: str, **fields) -> MagicMock:
    doc = MagicMock()
    # Redis search returns id as "prefix:id"
    doc.id = f"medha:{COLL}:{id_}"
    defaults = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "response_summary": "",
        "template_id": "",
        "usage_count": "1",
        "created_at": "0",
        "expires_at": "0",
        "__score": "0.05",
    }
    defaults.update(fields)
    for k, v in defaults.items():
        setattr(doc, k, v)
    return doc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ft():
    ft = AsyncMock()
    ft.info = AsyncMock(side_effect=Exception("unknown index name"))
    ft.create_index = AsyncMock()
    ft.search = AsyncMock(return_value=MagicMock(docs=[]))
    ft.dropindex = AsyncMock()
    return ft


@pytest.fixture
def mock_pipe():
    pipe = MagicMock()
    pipe.hset = MagicMock()
    pipe.delete = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    return pipe


@pytest.fixture
def mock_redis_client(mock_ft, mock_pipe):
    client = AsyncMock()
    client.ping = AsyncMock()
    client.ft = MagicMock(return_value=mock_ft)
    client.pipeline = MagicMock(return_value=mock_pipe)
    client.hexists = AsyncMock(return_value=True)
    client.hincrby = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
async def redis_backend(mock_redis_client, mock_ft, mock_pipe):
    from medha.backends.redis_vector import RedisVectorBackend

    with patch("medha.backends.redis_vector.Redis", return_value=mock_redis_client):
        b = RedisVectorBackend(_redis_settings())
        await b.connect()
        await b.initialize(COLL, DIM)

    # Reset info side_effect so subsequent count() calls succeed
    mock_ft.info.side_effect = None
    mock_ft.info.return_value = {"num_docs": 0}

    yield b, mock_ft, mock_pipe, mock_redis_client


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_creates_redis_client(mock_redis_client):
    from medha.backends.redis_vector import RedisVectorBackend

    with patch("medha.backends.redis_vector.Redis", return_value=mock_redis_client) as mock_cls:
        b = RedisVectorBackend(_redis_settings())
        await b.connect()
        await b.close()

    mock_cls.assert_called_once()
    mock_redis_client.ping.assert_awaited_once()


async def test_connect_uses_url(mock_redis_client):
    from medha.backends.redis_vector import RedisVectorBackend

    settings = _redis_settings(redis_url="redis://localhost:6379/0")
    with patch("medha.backends.redis_vector.aioredis.from_url", return_value=mock_redis_client) as mock_from_url:
        b = RedisVectorBackend(settings)
        await b.connect()
        await b.close()

    mock_from_url.assert_called_once()
    assert "redis://localhost:6379/0" in str(mock_from_url.call_args)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_creates_index_when_not_exists(mock_redis_client, mock_ft):
    from medha.backends.redis_vector import RedisVectorBackend

    mock_ft.info.side_effect = Exception("unknown index name")

    with patch("medha.backends.redis_vector.Redis", return_value=mock_redis_client):
        b = RedisVectorBackend(_redis_settings())
        await b.connect()
        await b.initialize(COLL, DIM)

    mock_ft.create_index.assert_awaited_once()


async def test_initialize_skips_when_index_exists(mock_redis_client, mock_ft):
    from medha.backends.redis_vector import RedisVectorBackend

    mock_ft.info.side_effect = None
    mock_ft.info.return_value = {"num_docs": 0}

    with patch("medha.backends.redis_vector.Redis", return_value=mock_redis_client):
        b = RedisVectorBackend(_redis_settings())
        await b.connect()
        await b.initialize(COLL, DIM)

    mock_ft.create_index.assert_not_awaited()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_pipeline(redis_backend):
    b, ft, pipe, client = redis_backend
    entries = [_make_entry() for _ in range(3)]

    await b.upsert(COLL, entries)

    assert pipe.hset.call_count == 3
    pipe.execute.assert_awaited_once()


async def test_upsert_empty_list_no_pipeline(redis_backend):
    b, ft, pipe, client = redis_backend

    await b.upsert(COLL, [])

    client.pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_calls_ft_search(redis_backend):
    b, ft, pipe, client = redis_backend

    results = await b.search(COLL, [0.1] * DIM, limit=5)

    ft.search.assert_awaited()
    assert results == []


async def test_search_returns_results(redis_backend):
    b, ft, pipe, client = redis_backend
    eid = str(uuid.uuid4())
    doc = _make_doc(eid, __score="0.05")  # score = 1 - 0.05 = 0.95
    ft.search.return_value = MagicMock(docs=[doc])

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.95, abs=1e-5)


async def test_search_score_threshold_filters(redis_backend):
    b, ft, pipe, client = redis_backend
    eid = str(uuid.uuid4())
    doc = _make_doc(eid, __score="0.9")  # score = 1 - 0.9 = 0.1
    ft.search.return_value = MagicMock(docs=[doc])

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.5)

    assert results == []


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------


async def test_scroll_returns_all(redis_backend):
    b, ft, pipe, client = redis_backend
    docs = [_make_doc(str(uuid.uuid4())) for _ in range(3)]
    ft.search.return_value = MagicMock(docs=docs)

    results, next_offset = await b.scroll(COLL, limit=10)

    assert len(results) == 3
    assert next_offset is None


async def test_scroll_pagination(redis_backend):
    b, ft, pipe, client = redis_backend
    docs = [_make_doc(str(uuid.uuid4())) for _ in range(2)]
    ft.search.return_value = MagicMock(docs=docs)

    results, next_offset = await b.scroll(COLL, limit=2)

    assert len(results) == 2
    assert next_offset == "2"


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_returns_num_docs(redis_backend):
    b, ft, pipe, client = redis_backend
    ft.info.return_value = {"num_docs": 7}

    result = await b.count(COLL)

    assert result == 7


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_calls_pipeline(redis_backend):
    b, ft, pipe, client = redis_backend
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    await b.delete(COLL, ids)

    assert pipe.delete.call_count == 2
    pipe.execute.assert_awaited()


async def test_delete_empty_list_no_pipeline(redis_backend):
    b, ft, pipe, client = redis_backend

    await b.delete(COLL, [])

    client.pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# search_by_query_hash
# ---------------------------------------------------------------------------


async def test_search_by_query_hash_found(redis_backend):
    b, ft, pipe, client = redis_backend
    eid = str(uuid.uuid4())
    doc = _make_doc(eid, query_hash="abc123")
    ft.search.return_value = MagicMock(docs=[doc])

    result = await b.search_by_query_hash(COLL, "abc123")

    assert result is not None
    assert result.id == eid


async def test_search_by_query_hash_not_found(redis_backend):
    b, ft, pipe, client = redis_backend
    ft.search.return_value = MagicMock(docs=[])

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count
# ---------------------------------------------------------------------------


async def test_update_usage_count_calls_hincrby(redis_backend):
    b, ft, pipe, client = redis_backend
    eid = str(uuid.uuid4())
    client.hexists.return_value = True

    await b.update_usage_count(COLL, eid)

    client.hincrby.assert_awaited_once()
    hincrby_args = client.hincrby.call_args.args
    assert "usage_count" in hincrby_args
    assert hincrby_args[-1] == 1


async def test_update_usage_count_unknown_id(redis_backend):
    b, ft, pipe, client = redis_backend
    client.hexists.return_value = False

    await b.update_usage_count(COLL, "nonexistent")  # must not raise

    client.hincrby.assert_not_awaited()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_calls_aclose(mock_redis_client, mock_ft, mock_pipe):
    from medha.backends.redis_vector import RedisVectorBackend

    mock_ft.info.side_effect = Exception("unknown index name")

    with patch("medha.backends.redis_vector.Redis", return_value=mock_redis_client):
        b = RedisVectorBackend(_redis_settings())
        await b.connect()
        assert b._client is not None
        await b.close()

    mock_redis_client.aclose.assert_awaited_once()
    assert b._client is None


# ---------------------------------------------------------------------------
# not connected guard
# ---------------------------------------------------------------------------


async def test_not_connected_raises():
    from medha.backends.redis_vector import RedisVectorBackend

    b = RedisVectorBackend.__new__(RedisVectorBackend)
    b._client = None
    b._settings = _redis_settings()

    with pytest.raises(StorageError, match="connect()"):
        await b.initialize(COLL, DIM)

    with pytest.raises(StorageError, match="connect()"):
        await b.search(COLL, [0.1] * DIM)

    with pytest.raises(StorageError, match="connect()"):
        await b.upsert(COLL, [_make_entry()])

    with pytest.raises(StorageError, match="connect()"):
        await b.count(COLL)

    with pytest.raises(StorageError, match="connect()"):
        await b.delete(COLL, ["some-id"])

    with pytest.raises(StorageError, match="connect()"):
        await b.search_by_query_hash(COLL, "hash")

    with pytest.raises(StorageError, match="connect()"):
        await b.update_usage_count(COLL, str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# missing deps
# ---------------------------------------------------------------------------


async def test_missing_deps_raises():
    with patch("medha.backends.redis_vector.HAS_REDIS", False):
        from medha.backends.redis_vector import RedisVectorBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            RedisVectorBackend()


# ---------------------------------------------------------------------------
# find_expired
# ---------------------------------------------------------------------------


async def test_find_expired_queries_range(redis_backend):
    b, ft, pipe, client = redis_backend
    ft.search.return_value = MagicMock(docs=[])

    expired_ids = await b.find_expired(COLL)

    ft.search.assert_awaited()
    assert expired_ids == []


async def test_find_expired_returns_ids(redis_backend):
    b, ft, pipe, client = redis_backend
    eid = str(uuid.uuid4())
    doc = _make_doc(eid)
    ft.search.return_value = MagicMock(docs=[doc])

    expired_ids = await b.find_expired(COLL)

    assert len(expired_ids) == 1
    assert expired_ids[0] == eid


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


async def test_drop_collection_calls_dropindex(redis_backend):
    b, ft, pipe, client = redis_backend

    await b.drop_collection(COLL)

    ft.dropindex.assert_awaited_once()


# ---------------------------------------------------------------------------
# find_by_query_hash
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(redis_backend):
    b, ft, pipe, client = redis_backend
    eid = str(uuid.uuid4())
    doc = _make_doc(eid, query_hash="abc123")
    ft.search.return_value = MagicMock(docs=[doc])

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert len(ids) == 1
    assert ids[0] == eid


async def test_find_by_query_hash_empty(redis_backend):
    b, ft, pipe, client = redis_backend
    ft.search.return_value = MagicMock(docs=[])

    ids = await b.find_by_query_hash(COLL, "nonexistent")

    assert ids == []


# ---------------------------------------------------------------------------
# find_by_template_id
# ---------------------------------------------------------------------------


async def test_find_by_template_id(redis_backend):
    b, ft, pipe, client = redis_backend
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    docs = [_make_doc(e, template_id="tmpl1") for e in eids]
    ft.search.return_value = MagicMock(docs=docs)

    ids = await b.find_by_template_id(COLL, "tmpl1")

    assert set(ids) == set(eids)


# ---------------------------------------------------------------------------
# search_by_normalized_question
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(redis_backend):
    b, ft, pipe, client = redis_backend
    eid = str(uuid.uuid4())
    doc = _make_doc(eid, normalized_question="how many users")
    ft.search.return_value = MagicMock(docs=[doc])

    result = await b.search_by_normalized_question(COLL, "how many users")

    assert result is not None
    assert result.id == eid


async def test_search_by_normalized_question_not_found(redis_backend):
    b, ft, pipe, client = redis_backend
    ft.search.return_value = MagicMock(docs=[])

    result = await b.search_by_normalized_question(COLL, "nothing")

    assert result is None
