"""Unit tests for medha.interfaces.storage.VectorStorageBackend ABC."""

import pytest

from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheResult


class TestVectorStorageBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            VectorStorageBackend()

    def test_partial_implementation_fails(self):
        class PartialBackend(VectorStorageBackend):
            async def initialize(self, collection_name, dimension, **kwargs):
                pass

            # Missing search, upsert, scroll, count, delete, close

        with pytest.raises(TypeError):
            PartialBackend()


# ---------------------------------------------------------------------------
# Cross-backend contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("any_backend", ["memory"], indirect=True)
class TestBackendContract:
    """These tests must pass on ALL backends."""

    async def test_initialize_is_idempotent(self, any_backend):
        await any_backend.initialize("contract_test", 8)
        await any_backend.initialize("contract_test", 8)  # no exception

    async def test_upsert_and_count(self, any_backend, make_entry_fixture):
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [make_entry_fixture()])
        assert await any_backend.count("contract_test") == 1

    async def test_search_returns_cache_result(self, any_backend, make_entry_fixture):
        vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        entry = make_entry_fixture(vector=vec, dim=8)
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])
        results = await any_backend.search("contract_test", vec, limit=1)
        assert len(results) == 1
        assert isinstance(results[0], CacheResult)
        assert results[0].score > 0.9

    async def test_delete_removes_entry(self, any_backend, make_entry_fixture):
        entry = make_entry_fixture()
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])
        await any_backend.delete("contract_test", [entry.id])
        assert await any_backend.count("contract_test") == 0

    async def test_scroll_returns_all(self, any_backend, make_entry_fixture):
        entries = [make_entry_fixture() for _ in range(5)]
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", entries)
        results, next_offset = await any_backend.scroll("contract_test", limit=10)
        assert len(results) == 5
        assert next_offset is None
