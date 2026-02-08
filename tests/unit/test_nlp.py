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
