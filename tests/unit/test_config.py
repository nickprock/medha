"""Unit tests for medha.config.Settings."""

import pytest
from pydantic import ValidationError

from medha.config import Settings


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.qdrant_mode == "memory"
        assert s.score_threshold_exact == 0.99
        assert s.score_threshold_semantic == 0.90
        assert s.l1_cache_max_size == 1000

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            Settings(qdrant_mode="invalid")

    def test_exact_threshold_too_low(self):
        with pytest.raises(ValidationError):
            Settings(score_threshold_exact=0.5)

    def test_semantic_must_be_below_exact(self):
        with pytest.raises(ValidationError):
            Settings(score_threshold_exact=0.99, score_threshold_semantic=0.99)

    def test_env_var_loading(self, monkeypatch):
        monkeypatch.setenv("MEDHA_QDRANT_MODE", "docker")
        s = Settings()
        assert s.qdrant_mode == "docker"

    def test_settings_dump(self):
        s = Settings()
        d = s.model_dump()
        assert isinstance(d, dict)
        assert "qdrant_mode" in d
        assert "score_threshold_exact" in d

    def test_fuzzy_prefilter_defaults(self):
        s = Settings()
        assert s.score_threshold_fuzzy_prefilter == 0.65
        assert s.fuzzy_prefilter_top_k == 50

    def test_fuzzy_prefilter_threshold_bounds(self):
        with pytest.raises(ValidationError):
            Settings(score_threshold_fuzzy_prefilter=1.5)
        with pytest.raises(ValidationError):
            Settings(score_threshold_fuzzy_prefilter=-0.1)

    def test_fuzzy_prefilter_top_k_bounds(self):
        with pytest.raises(ValidationError):
            Settings(fuzzy_prefilter_top_k=0)
        with pytest.raises(ValidationError):
            Settings(fuzzy_prefilter_top_k=1001)

    def test_fuzzy_prefilter_custom_values(self):
        s = Settings(score_threshold_fuzzy_prefilter=0.50, fuzzy_prefilter_top_k=100)
        assert s.score_threshold_fuzzy_prefilter == 0.50
        assert s.fuzzy_prefilter_top_k == 100


class TestBackendTypeSettings:
    def test_backend_type_default(self):
        s = Settings()
        assert s.backend_type == "qdrant"

    def test_backend_type_memory(self):
        s = Settings(backend_type="memory")
        assert s.backend_type == "memory"

    def test_backend_type_pgvector(self):
        s = Settings(backend_type="pgvector")
        assert s.backend_type == "pgvector"

    def test_backend_type_invalid(self):
        with pytest.raises(ValidationError):
            Settings(backend_type="unknown")

    def test_pg_pool_max_lt_min_raises(self):
        with pytest.raises(ValidationError):
            Settings(backend_type="pgvector", pg_pool_min_size=5, pg_pool_max_size=2)

    def test_pg_dsn_overrides_fields(self):
        # DSN è una stringa, non viene parsata — solo passata ad asyncpg
        s = Settings(pg_dsn="postgresql://user:pass@host/db")
        assert s.pg_dsn == "postgresql://user:pass@host/db"
