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
