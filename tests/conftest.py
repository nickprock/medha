"""Shared pytest fixtures for Medha tests."""

import hashlib
from typing import List

import pytest

from medha.config import Settings
from medha.interfaces.embedder import BaseEmbedder
from medha.types import QueryTemplate


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder for unit tests.

    Generates vectors by hashing the input text and spreading the hash
    across the requested dimension. Identical inputs always produce
    identical vectors; similar inputs produce similar (but not identical)
    vectors.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._model_name = "mock-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def aembed(self, text: str) -> List[float]:
        """Generate a deterministic embedding from text hash."""
        h = hashlib.sha256(text.lower().encode()).hexdigest()
        # Expand hash to fill dimension
        values = []
        for i in range(self._dimension):
            byte_val = int(h[(i * 2) % len(h) : (i * 2 + 2) % len(h) or len(h)], 16)
            values.append((byte_val / 255.0) * 2 - 1)  # Normalize to [-1, 1]
        # Normalize to unit vector
        magnitude = sum(v**2 for v in values) ** 0.5
        return [v / magnitude for v in values] if magnitude > 0 else values

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.aembed(t) for t in texts]


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder with dimension=384."""
    return MockEmbedder(dimension=384)


@pytest.fixture
def test_settings():
    """Provide test-friendly settings (in-memory Qdrant, relaxed thresholds)."""
    return Settings(
        qdrant_mode="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


@pytest.fixture
def sample_templates():
    """Provide sample QueryTemplate objects for testing."""
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
