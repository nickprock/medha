"""Unit tests for medha.exceptions hierarchy."""

import pytest

from medha.exceptions import (
    MedhaError,
    ConfigurationError,
    EmbeddingError,
    StorageError,
    StorageInitializationError,
    TemplateError,
    ParameterExtractionError,
)


class TestExceptionHierarchy:
    def test_all_catchable_via_medha_error(self):
        exceptions = [
            ConfigurationError("cfg"),
            EmbeddingError("emb"),
            StorageError("store"),
            StorageInitializationError("init"),
            TemplateError("tpl"),
            ParameterExtractionError("param"),
        ]
        for exc in exceptions:
            with pytest.raises(MedhaError):
                raise exc

    def test_storage_initialization_is_storage_error(self):
        with pytest.raises(StorageError):
            raise StorageInitializationError("init failed")

    def test_medha_error_is_exception(self):
        with pytest.raises(Exception):
            raise MedhaError("base")

    def test_exception_messages(self):
        exc = ConfigurationError("bad config")
        assert str(exc) == "bad config"

    def test_inheritance_chain(self):
        assert issubclass(ConfigurationError, MedhaError)
        assert issubclass(EmbeddingError, MedhaError)
        assert issubclass(StorageError, MedhaError)
        assert issubclass(StorageInitializationError, StorageError)
        assert issubclass(StorageInitializationError, MedhaError)
        assert issubclass(TemplateError, MedhaError)
        assert issubclass(ParameterExtractionError, MedhaError)
