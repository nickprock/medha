"""Unit tests for WeaviateBackend (mocked weaviate async client — no real server required)."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

weaviate = pytest.importorskip("weaviate")

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


def _weaviate_settings(**overrides) -> Settings:
    return Settings(backend_type="weaviate", **overrides)


def _make_wv_obj(id_: str, **props) -> MagicMock:
    obj = MagicMock()
    obj.uuid = uuid.UUID(id_)
    default_props = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "usage_count": 1,
        "created_at": None,
        "expires_at": None,
        "response_summary": "",
        "template_id": "",
    }
    default_props.update(props)
    obj.properties = default_props
    obj.metadata = MagicMock()
    obj.metadata.distance = 0.05
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_wv_collection():
    col = AsyncMock()

    near_vector_result = MagicMock()
    near_vector_result.objects = []
    col.query = AsyncMock()
    col.query.near_vector = AsyncMock(return_value=near_vector_result)
    col.query.fetch_objects = AsyncMock(return_value=MagicMock(objects=[]))
    col.query.fetch_object_by_id = AsyncMock(return_value=None)

    insert_result = MagicMock()
    insert_result.has_errors = False
    col.data = AsyncMock()
    col.data.insert_many = AsyncMock(return_value=insert_result)
    col.data.delete_by_id = AsyncMock()
    col.data.delete_many = AsyncMock()
    col.data.update = AsyncMock()

    agg_result = MagicMock()
    agg_result.total_count = 0
    col.aggregate = AsyncMock()
    col.aggregate.over_all = AsyncMock(return_value=agg_result)

    return col


@pytest.fixture
def mock_wv_client(mock_wv_collection):
    client = AsyncMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()

    collections_ns = AsyncMock()
    collections_ns.exists = AsyncMock(return_value=False)
    collections_ns.create = AsyncMock()
    collections_ns.delete = AsyncMock()
    collections_ns.get = MagicMock(return_value=mock_wv_collection)  # sync call

    client.collections = collections_ns
    return client


@pytest.fixture
async def wv_backend(mock_wv_client, mock_wv_collection):
    from medha.backends.weaviate import WeaviateBackend

    with patch("medha.backends.weaviate.weaviate.use_async_with_local", return_value=mock_wv_client):
        b = WeaviateBackend(_weaviate_settings())
        await b.connect()
        yield b, mock_wv_collection, mock_wv_client


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_calls_client_connect(mock_wv_client):
    from medha.backends.weaviate import WeaviateBackend

    with patch("medha.backends.weaviate.weaviate.use_async_with_local", return_value=mock_wv_client):
        b = WeaviateBackend(_weaviate_settings())
        await b.connect()

    mock_wv_client.connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_creates_collection(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)

    client.collections.exists.assert_awaited_once()
    client.collections.create.assert_awaited_once()
    assert COLL in b._collections


async def test_initialize_skips_when_exists(wv_backend):
    b, col, client = wv_backend
    client.collections.exists.return_value = True

    await b.initialize(COLL, DIM)

    client.collections.create.assert_not_awaited()


async def test_initialize_idempotent(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.exists.reset_mock()
    client.collections.create.reset_mock()

    await b.initialize(COLL, DIM)

    client.collections.exists.assert_not_awaited()
    client.collections.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_insert_many(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    entries = [_make_entry() for _ in range(3)]

    await b.upsert(COLL, entries)

    col.data.insert_many.assert_awaited_once()
    call_args = col.data.insert_many.call_args.args[0]
    assert len(call_args) == 3


async def test_upsert_empty_list_no_call(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)

    await b.upsert(COLL, [])

    col.data.insert_many.assert_not_awaited()


async def test_upsert_uninitialized_raises(wv_backend):
    b, _, _ = wv_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.upsert("nonexistent", [_make_entry()])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_calls_near_vector(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    col.query.near_vector.return_value = MagicMock(objects=[obj])

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    col.query.near_vector.assert_awaited_once()
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.95, abs=1e-5)  # 1.0 - 0.05


async def test_search_score_threshold(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    obj.metadata.distance = 0.9  # score = 1 - 0.9 = 0.1
    col.query.near_vector.return_value = MagicMock(objects=[obj])

    results = await b.search(COLL, [0.1] * DIM, score_threshold=0.5)

    assert results == []


async def test_search_uninitialized_raises(wv_backend):
    b, _, _ = wv_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.search("nonexistent", [0.1] * DIM)


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------


async def test_scroll_returns_entries(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    objects = [_make_wv_obj(str(uuid.uuid4())) for _ in range(3)]
    col.query.fetch_objects.return_value = MagicMock(objects=objects)

    results, next_offset = await b.scroll(COLL, limit=10)

    assert len(results) == 3
    assert next_offset is None  # 3 < limit=10


async def test_scroll_pagination(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    objects = [_make_wv_obj(str(uuid.uuid4())) for _ in range(2)]
    col.query.fetch_objects.return_value = MagicMock(objects=objects)

    results, next_offset = await b.scroll(COLL, limit=2)

    assert len(results) == 2
    assert next_offset == str(objects[-1].uuid)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_returns_value(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.aggregate.over_all.return_value = MagicMock(total_count=5)

    result = await b.count(COLL)

    assert result == 5


async def test_count_uninitialized_raises(wv_backend):
    b, _, _ = wv_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.count("nonexistent")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_few_ids_uses_delete_by_id(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()) for _ in range(3)]

    await b.delete(COLL, ids)

    assert col.data.delete_by_id.await_count == 3
    col.data.delete_many.assert_not_awaited()


async def test_delete_many_ids_uses_delete_many(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()) for _ in range(15)]

    await b.delete(COLL, ids)

    col.data.delete_many.assert_awaited_once()
    col.data.delete_by_id.assert_not_awaited()


async def test_delete_empty_list_no_call(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)

    await b.delete(COLL, [])

    col.data.delete_by_id.assert_not_awaited()
    col.data.delete_many.assert_not_awaited()


# ---------------------------------------------------------------------------
# search_by_query_hash
# ---------------------------------------------------------------------------


async def test_search_by_query_hash_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid, generated_query="SELECT 42")
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    result = await b.search_by_query_hash(COLL, "abc123")

    assert result is not None
    assert result.generated_query == "SELECT 42"


async def test_search_by_query_hash_not_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count
# ---------------------------------------------------------------------------


async def test_update_usage_count_increments(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid, usage_count=2)
    col.query.fetch_object_by_id.return_value = obj

    await b.update_usage_count(COLL, eid)

    col.data.update.assert_awaited_once()
    update_kwargs = col.data.update.call_args.kwargs
    assert update_kwargs["properties"]["usage_count"] == 3


async def test_update_usage_count_unknown_id(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_object_by_id.return_value = None

    await b.update_usage_count(COLL, "nonexistent")  # must not raise

    col.data.update.assert_not_awaited()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_calls_client_close(mock_wv_client, mock_wv_collection):
    from medha.backends.weaviate import WeaviateBackend

    with patch("medha.backends.weaviate.weaviate.use_async_with_local", return_value=mock_wv_client):
        b = WeaviateBackend(_weaviate_settings())
        await b.connect()
        await b.close()

    mock_wv_client.close.assert_awaited_once()
    assert b._client is None
    assert b._collections == {}


# ---------------------------------------------------------------------------
# missing deps
# ---------------------------------------------------------------------------


async def test_missing_deps_raises():
    with patch("medha.backends.weaviate.HAS_WEAVIATE", False):
        from medha.backends.weaviate import WeaviateBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            WeaviateBackend()


# ---------------------------------------------------------------------------
# find_expired
# ---------------------------------------------------------------------------


async def test_find_expired_returns_ids(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    expired_ids = await b.find_expired(COLL)

    col.query.fetch_objects.assert_awaited()
    assert expired_ids == [str(obj.uuid)]


async def test_find_expired_empty(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    expired_ids = await b.find_expired(COLL)

    assert expired_ids == []


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


async def test_drop_collection(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)

    await b.drop_collection(COLL)

    client.collections.delete.assert_awaited_once()
    assert COLL not in b._collections


async def test_drop_collection_unconnected_raises(wv_backend):
    b, col, _ = wv_backend
    b._client = None
    with pytest.raises(StorageError):
        await b.drop_collection(COLL)


# ---------------------------------------------------------------------------
# find_by_query_hash
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [str(obj.uuid)]


async def test_find_by_query_hash_empty(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    ids = await b.find_by_query_hash(COLL, "nonexistent")

    assert ids == []


# ---------------------------------------------------------------------------
# find_by_template_id
# ---------------------------------------------------------------------------


async def test_find_by_template_id(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    objects = [_make_wv_obj(e) for e in eids]
    col.query.fetch_objects.return_value = MagicMock(objects=objects)

    ids = await b.find_by_template_id(COLL, "tmpl1")

    assert set(ids) == {str(obj.uuid) for obj in objects}


# ---------------------------------------------------------------------------
# search_by_normalized_question
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid, generated_query="SELECT COUNT(*) FROM users")
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    result = await b.search_by_normalized_question(COLL, "how many users")

    assert result is not None
    assert result.generated_query == "SELECT COUNT(*) FROM users"


async def test_search_by_normalized_question_not_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    result = await b.search_by_normalized_question(COLL, "nothing")

    assert result is None
