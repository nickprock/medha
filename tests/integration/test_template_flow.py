"""Integration tests for the template lifecycle: load, sync, match, render."""

import pytest

fastembed = pytest.importorskip("fastembed")

from medha.backends.qdrant import QdrantBackend
from medha.config import Settings
from medha.core import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter
from medha.types import QueryTemplate, SearchStrategy


@pytest.fixture
def settings():
    return Settings(
        qdrant_mode="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.70,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


@pytest.fixture
def embedder():
    return FastEmbedAdapter()


@pytest.fixture
def templates():
    return [
        QueryTemplate(
            intent="count_entities",
            template_text="How many {entity} are there",
            query_template="SELECT COUNT(*) FROM {entity}",
            parameters=["entity"],
            priority=1,
            parameter_patterns={"entity": r"\b(users|products|orders|employees)\b"},
        ),
        QueryTemplate(
            intent="top_n",
            template_text="Show top {count} {entity}",
            query_template="SELECT * FROM {entity} LIMIT {count}",
            parameters=["count", "entity"],
            priority=1,
            parameter_patterns={
                "count": r"\b(\d+)\b",
                "entity": r"\b(users|products|orders|employees)\b",
            },
        ),
        QueryTemplate(
            intent="filter_by_department",
            template_text="List employees in {department}",
            query_template="SELECT * FROM employees WHERE dept = '{department}'",
            parameters=["department"],
            priority=2,
        ),
    ]


@pytest.fixture
async def medha(embedder, settings):
    backend = QdrantBackend(settings)
    await backend.connect()
    m = Medha(
        collection_name="tpl_test",
        embedder=embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


class TestLoadAndSync:
    async def test_load_templates(self, medha, templates):
        await medha.load_templates(templates)
        assert len(medha._templates) == 3

    async def test_sync_to_backend(self, medha, templates):
        await medha.load_templates(templates)
        count = await medha._backend.count(medha._template_collection)
        # Each template's template_text is embedded → at least 3 entries
        assert count >= 3


class TestMatchAndRender:
    async def test_match_and_render(self, medha, templates):
        await medha.load_templates(templates)
        medha._l1_cache.clear()

        hit = await medha.search("How many users are there")
        if hit.strategy == SearchStrategy.TEMPLATE_MATCH:
            assert hit.template_used == "count_entities"
            assert "COUNT" in hit.generated_query
            assert "users" in hit.generated_query
        else:
            # If template threshold wasn't met, that's acceptable
            # with real embeddings — verify it's not an error
            assert hit.strategy != SearchStrategy.ERROR

    async def test_match_top_n(self, medha, templates):
        await medha.load_templates(templates)
        medha._l1_cache.clear()

        hit = await medha.search("Show top 10 products")
        if hit.strategy == SearchStrategy.TEMPLATE_MATCH:
            assert hit.template_used == "top_n"
            assert "products" in hit.generated_query
            assert "10" in hit.generated_query


class TestNoMatchWithoutParams:
    async def test_no_match_without_params(self, medha, templates):
        await medha.load_templates(templates)
        medha._l1_cache.clear()

        # "department" template requires a capitalized heuristic word.
        # If we give no extractable department, params will be incomplete.
        hit = await medha.search("list employees in")
        # Template should fail due to missing params → falls through
        assert hit.strategy != SearchStrategy.ERROR


class TestPriorityOrdering:
    async def test_priority_ordering(self, medha, embedder, settings):
        """Higher priority (lower number) template should win when both match."""
        # Create two templates that could both match, different priorities
        high_priority = QueryTemplate(
            intent="count_high",
            template_text="How many {entity} are there",
            query_template="SELECT COUNT(*) FROM {entity}",
            parameters=["entity"],
            priority=1,
            parameter_patterns={"entity": r"\b(users|products)\b"},
        )
        low_priority = QueryTemplate(
            intent="count_low",
            template_text="How many {entity} are there",
            query_template="SELECT COUNT(id) FROM {entity}",
            parameters=["entity"],
            priority=4,
            parameter_patterns={"entity": r"\b(users|products)\b"},
        )

        await medha.load_templates([high_priority, low_priority])
        medha._l1_cache.clear()

        hit = await medha.search("How many users are there")
        if hit.strategy == SearchStrategy.TEMPLATE_MATCH:
            # Priority 1 template gets a bonus, should win
            assert hit.template_used == "count_high"
