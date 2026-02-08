"""Unit tests for medha.interfaces.storage.VectorStorageBackend ABC."""

import pytest

from medha.interfaces.storage import VectorStorageBackend


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
