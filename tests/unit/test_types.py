"""Unit tests for medha.types data models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from medha.types import CacheHit, CacheEntry, CacheResult, QueryTemplate, SearchStrategy


class TestCacheHit:
    def test_cache_hit_defaults(self):
        hit = CacheHit()
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert hit.confidence == 0.0
        assert hit.generated_query == ""
        assert hit.response_summary is None
        assert hit.template_used is None

    def test_cache_hit_valid(self):
        hit = CacheHit(
            generated_query="SELECT 1",
            response_summary="one row",
            confidence=0.95,
            strategy=SearchStrategy.SEMANTIC_MATCH,
            template_used="count_entities",
        )
        assert hit.generated_query == "SELECT 1"
        assert hit.response_summary == "one row"
        assert hit.confidence == 0.95
        assert hit.strategy == SearchStrategy.SEMANTIC_MATCH
        assert hit.template_used == "count_entities"

    def test_cache_hit_confidence_bounds(self):
        with pytest.raises(ValidationError):
            CacheHit(confidence=1.5)
        with pytest.raises(ValidationError):
            CacheHit(confidence=-0.1)

    def test_cache_hit_strategy_enum(self):
        hit = CacheHit(strategy="exact_match")
        assert hit.strategy == SearchStrategy.EXACT_MATCH

    def test_cache_hit_frozen(self):
        hit = CacheHit()
        with pytest.raises(ValidationError):
            hit.confidence = 0.5


class TestQueryTemplate:
    def test_query_template_roundtrip(self):
        template = QueryTemplate(
            intent="count_entities",
            template_text="How many {entity} are there",
            query_template="SELECT COUNT(*) FROM {entity}",
            parameters=["entity"],
            priority=1,
            parameter_patterns={"entity": r"\b(users|products)\b"},
        )
        json_str = template.model_dump_json()
        restored = QueryTemplate.model_validate_json(json_str)
        assert restored.intent == template.intent
        assert restored.template_text == template.template_text
        assert restored.query_template == template.query_template
        assert restored.parameters == template.parameters
        assert restored.priority == template.priority
        assert restored.parameter_patterns == template.parameter_patterns

    def test_query_template_defaults(self):
        template = QueryTemplate(
            intent="test",
            template_text="test {x}",
            query_template="SELECT {x}",
        )
        assert template.parameters == []
        assert template.aliases == []
        assert template.parameter_patterns == {}
        assert template.priority == 1


class TestCacheEntry:
    def test_cache_entry_auto_timestamp(self):
        before = datetime.now(timezone.utc)
        entry = CacheEntry(
            id="abc-123",
            vector=[0.1, 0.2],
            original_question="test",
            normalized_question="test",
            generated_query="SELECT 1",
            query_hash="d41d8cd98f00b204e9800998ecf8427e",
        )
        after = datetime.now(timezone.utc)
        assert before <= entry.created_at <= after

    def test_cache_entry_query_hash(self):
        import hashlib

        query = "SELECT COUNT(*) FROM users"
        expected = hashlib.md5(query.strip().encode("utf-8")).hexdigest()
        entry = CacheEntry(
            id="abc-123",
            vector=[0.1],
            original_question="q",
            normalized_question="q",
            generated_query=query,
            query_hash=expected,
        )
        assert entry.query_hash == expected
        assert len(entry.query_hash) == 32
