"""Unit tests for medha.utils.nlp."""

import pytest

from medha.types import QueryTemplate
from medha.utils.nlp import ParameterExtractor, keyword_overlap_score
from medha.exceptions import ParameterExtractionError


@pytest.fixture
def extractor():
    return ParameterExtractor(use_spacy=False)


@pytest.fixture
def count_template():
    return QueryTemplate(
        intent="top_n",
        template_text="Show top {count} {entity}",
        query_template="SELECT * FROM {entity} LIMIT {count}",
        parameters=["count", "entity"],
        priority=1,
        parameter_patterns={
            "count": r"\b(\d+)\b",
            "entity": r"\b(users|products|orders|employees)\b",
        },
    )


@pytest.fixture
def dept_template():
    return QueryTemplate(
        intent="filter_by_department",
        template_text="List employees in {department}",
        query_template="SELECT * FROM employees WHERE dept = '{department}'",
        parameters=["department"],
        priority=2,
    )


class TestExtractViaRegex:
    def test_extract_number_via_regex(self, extractor, count_template):
        params = extractor.extract("Show top 5 products", count_template)
        assert params["count"] == "5"

    def test_extract_entity_via_regex(self, extractor, count_template):
        params = extractor.extract("Show top 5 users", count_template)
        assert params["entity"] == "users"


class TestExtractViaHeuristics:
    def test_extract_via_heuristics(self, extractor, dept_template):
        # Use lowercase prefix so only "Engineering" is a capitalized word
        params = extractor.extract("list employees in Engineering", dept_template)
        assert params["department"] == "Engineering"


class TestRenderQuery:
    def test_render_query(self, extractor, count_template):
        params = {"count": "10", "entity": "products"}
        result = extractor.render_query(count_template, params)
        assert result == "SELECT * FROM products LIMIT 10"

    def test_render_query_unfilled_placeholder(self, extractor, count_template):
        with pytest.raises(ParameterExtractionError):
            extractor.render_query(count_template, {"count": "10"})


class TestSanitizeValue:
    def test_sanitize_value(self):
        assert ParameterExtractor._sanitize_value("hello") == "hello"
        assert ParameterExtractor._sanitize_value("it's") == "its"
        assert ParameterExtractor._sanitize_value("DROP;--") == "DROP--"


class TestKeywordOverlapScore:
    def test_keyword_overlap_score(self):
        score = keyword_overlap_score("show top employees", "Show top {count} {entity}")
        assert score > 0

    def test_no_overlap(self):
        score = keyword_overlap_score("unrelated xyz", "Show top {count} {entity}")
        assert score == 0.0

    def test_empty_template(self):
        score = keyword_overlap_score("anything", "")
        assert score == 0.0


class TestNoSpacyFallback:
    def test_no_spacy_fallback(self):
        ext = ParameterExtractor(use_spacy=False)
        assert ext.spacy_available is False
        # Should still work via regex + heuristics
        template = QueryTemplate(
            intent="top_n",
            template_text="Show top {count} {entity}",
            query_template="SELECT * FROM {entity} LIMIT {count}",
            parameters=["count", "entity"],
            parameter_patterns={
                "count": r"\b(\d+)\b",
                "entity": r"\b(users|products)\b",
            },
        )
        params = ext.extract("Show top 5 products", template)
        assert params == {"count": "5", "entity": "products"}


class TestExtractEntities:
    def test_extract_numbers(self, extractor):
        entities = extractor.extract_entities("Show top 5 products limit 10")
        assert "number" in entities
        assert "5" in entities["number"]
        assert "10" in entities["number"]

    def test_extract_capitalized_names(self, extractor):
        # Use lowercase "find" so only "John Smith" is capitalized
        entities = extractor.extract_entities("find John Smith in the database")
        assert "person" in entities
        assert "John Smith" in entities["person"]

    def test_extract_empty_text(self, extractor):
        entities = extractor.extract_entities("no entities here")
        assert "number" not in entities


class TestExtractNoParameters:
    def test_extract_no_params_required(self, extractor):
        template = QueryTemplate(
            intent="list_all",
            template_text="Show all records",
            query_template="SELECT * FROM records",
        )
        params = extractor.extract("Show all records", template)
        assert params == {}


class TestExtractionFailure:
    def test_missing_params_raises(self, extractor):
        template = QueryTemplate(
            intent="multi",
            template_text="{a} and {b}",
            query_template="SELECT {a}, {b}",
            parameters=["a", "b"],
        )
        with pytest.raises(ParameterExtractionError):
            extractor.extract("nothing matches here", template)


class TestEntityCache:
    def test_same_text_returns_cached_result(self, extractor):
        """extract_entities() on the same text must not re-parse."""
        call_count = 0
        original_findall = __import__("re").findall

        import re

        def counting_findall(pattern, string, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_findall(pattern, string, *args, **kwargs)

        import medha.utils.nlp as nlp_module
        original = nlp_module.re.findall
        nlp_module.re.findall = counting_findall

        try:
            extractor.extract_entities("find 5 users")
            calls_after_first = call_count
            call_count = 0
            extractor.extract_entities("find 5 users")  # should hit cache
            assert call_count == 0, "Cache miss: re.findall was called on repeated input"
        finally:
            nlp_module.re.findall = original

    def test_different_texts_are_cached_separately(self, extractor):
        r1 = extractor.extract_entities("show 5 records")
        r2 = extractor.extract_entities("show 10 records")
        assert r1 != r2
        assert "5" in r1.get("number", [])
        assert "10" in r2.get("number", [])

    def test_cache_is_populated_after_first_call(self, extractor):
        extractor._entity_cache.clear()
        extractor.extract_entities("unique question 42")
        assert "unique question 42" in extractor._entity_cache

    def test_cache_evicts_when_full(self, extractor):
        extractor._entity_cache.clear()
        # Fill cache to max
        for i in range(ParameterExtractor._ENTITY_CACHE_MAXSIZE):
            extractor.extract_entities(f"question number {i}")
        assert len(extractor._entity_cache) == ParameterExtractor._ENTITY_CACHE_MAXSIZE
        # One more entry triggers eviction (clear-on-full)
        extractor.extract_entities("overflow question")
        assert len(extractor._entity_cache) == 1

    def test_cache_hit_returns_same_object(self, extractor):
        extractor._entity_cache.clear()
        result1 = extractor.extract_entities("stable text 99")
        result2 = extractor.extract_entities("stable text 99")
        assert result1 == result2
