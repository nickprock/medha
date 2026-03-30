"""Edge case and error-path tests for Medha core."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from medha.config import Settings
from medha.core import Medha
from medha.exceptions import EmbeddingError
from medha.types import SearchStrategy
from tests.conftest import MockEmbedder
from tests.unit.test_core_waterfall import MockBackend

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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
def medha_instance(settings):
    embedder = MockEmbedder(dimension=384)
    return Medha(
        collection_name="edge_test",
        embedder=embedder,
        backend=MockBackend(),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Unicode and special characters
# ---------------------------------------------------------------------------

class TestUnicodeInputs:
    async def test_chinese_question_stores_and_retrieves(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("有多少用户", "SELECT COUNT(*) FROM users")
        assert ok is True
        hit = await medha_instance.search("有多少用户")
        assert hit.strategy != SearchStrategy.ERROR

    async def test_arabic_question_stores_and_retrieves(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("كم عدد المستخدمين", "SELECT COUNT(*) FROM users")
        assert ok is True
        hit = await medha_instance.search("كم عدد المستخدمين")
        assert hit.strategy != SearchStrategy.ERROR

    async def test_emoji_in_question(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("How many users? 🤔", "SELECT COUNT(*) FROM users")
        assert ok is True

    async def test_mixed_scripts_in_question(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("users: المستخدمون 用户", "SELECT * FROM users")
        assert ok is True

    async def test_question_with_sql_special_chars(self, medha_instance):
        """Questions with SQL-like content must not cause injection or failures."""
        await medha_instance.start()
        ok = await medha_instance.store(
            "users where name = 'Alice' OR 1=1",
            "SELECT * FROM users WHERE name = 'Alice'",
        )
        assert ok is True


# ---------------------------------------------------------------------------
# Long inputs
# ---------------------------------------------------------------------------

class TestLongInputs:
    async def test_very_long_question(self, medha_instance):
        await medha_instance.start()
        long_q = "How many " + "really " * 200 + "users are there"
        ok = await medha_instance.store(long_q, "SELECT COUNT(*) FROM users")
        assert ok is True
        hit = await medha_instance.search(long_q)
        assert hit.strategy != SearchStrategy.ERROR

    async def test_very_long_generated_query(self, medha_instance):
        await medha_instance.start()
        long_query = (
            "SELECT " + ", ".join(f"col_{i}" for i in range(100)) + " FROM users"
        )
        ok = await medha_instance.store("complex query", long_query)
        assert ok is True

    async def test_single_char_question(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("x", "SELECT 1")
        assert ok is True
        hit = await medha_instance.search("x")
        assert hit.strategy != SearchStrategy.ERROR

    async def test_repeated_whitespace_question(self, medha_instance):
        """Questions with extra whitespace should be normalized consistently."""
        await medha_instance.start()
        await medha_instance.store("How many users", "SELECT COUNT(*) FROM users")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()
        hit = await medha_instance.search("How  many   users")
        # Normalization collapses whitespace — should find the stored entry
        assert hit.strategy != SearchStrategy.ERROR


# ---------------------------------------------------------------------------
# Error paths — embedder failures
# ---------------------------------------------------------------------------

class TestEmbedderErrorPaths:
    async def test_search_returns_error_when_embedder_fails(self, settings):
        """If the embedder raises EmbeddingError, search must return ERROR strategy."""
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="emb_err_search",
            embedder=embedder,
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        await m.store("test question", "SELECT 1")
        await m._l1_backend.clear()
        m._embedding_cache.clear()

        embedder.aembed = AsyncMock(side_effect=EmbeddingError("mock failure"))
        hit = await m.search("test question")
        assert hit.strategy == SearchStrategy.ERROR

    async def test_store_returns_false_when_embedder_fails(self, settings):
        """If the embedder raises EmbeddingError during store, returns False."""
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="emb_err_store",
            embedder=embedder,
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        embedder.aembed = AsyncMock(side_effect=EmbeddingError("mock failure"))
        ok = await m.store("failing question", "SELECT 1")
        assert ok is False

    async def test_store_batch_returns_false_when_embedding_fails(self, settings):
        """store_batch returns False when aembed_batch raises EmbeddingError."""
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="emb_err_batch",
            embedder=embedder,
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        embedder.aembed_batch = AsyncMock(side_effect=EmbeddingError("mock batch failure"))
        ok = await m.store_batch([
            {"question": "q1", "generated_query": "SELECT 1"},
            {"question": "q2", "generated_query": "SELECT 2"},
        ])
        assert ok is False


# ---------------------------------------------------------------------------
# Batch edge cases
# ---------------------------------------------------------------------------

class TestBatchEdgeCases:
    async def test_store_batch_empty_list(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store_batch([])
        assert ok is True

    async def test_store_batch_single_item(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store_batch([
            {"question": "single item", "generated_query": "SELECT 1"}
        ])
        assert ok is True

    async def test_store_batch_with_all_optional_fields(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store_batch([
            {
                "question": "q with all fields",
                "generated_query": "SELECT 1",
                "response_summary": "one row",
                "template_id": "count",
            }
        ])
        assert ok is True

    async def test_store_batch_skips_empty_question(self, medha_instance):
        """Entries with empty question are filtered; valid ones are still stored."""
        await medha_instance.start()
        ok = await medha_instance.store_batch([
            {"question": "", "generated_query": "SELECT 1"},
            {"question": "valid question", "generated_query": "SELECT 2"},
        ])
        assert ok is True

    async def test_store_batch_all_invalid_returns_false(self, medha_instance):
        """If every entry is invalid, store_batch returns False."""
        await medha_instance.start()
        ok = await medha_instance.store_batch([
            {"question": "  ", "generated_query": "SELECT 1"},
            {"question": "q", "generated_query": ""},
        ])
        assert ok is False


# ---------------------------------------------------------------------------
# Embedding deduplication
# ---------------------------------------------------------------------------

class TestEmbeddingDeduplication:
    async def test_concurrent_same_question_computes_once(self, settings):
        """Multiple concurrent _get_embedding calls for the same key compute only once."""
        call_count = 0
        original_embedder = MockEmbedder(dimension=384)

        class CountingEmbedder(MockEmbedder):
            async def aembed(self, text: str):
                nonlocal call_count
                await asyncio.sleep(0)  # yield so other coroutines can start
                call_count += 1
                return await original_embedder.aembed(text)

        m = Medha(
            collection_name="dedup_test",
            embedder=CountingEmbedder(dimension=384),
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()

        # Fire 5 concurrent embedding requests for the identical question
        results = await asyncio.gather(*[
            m._get_embedding("same question for dedup") for _ in range(5)
        ])

        assert all(r is not None for r in results)
        # All results must be identical vectors
        assert all(r == results[0] for r in results)
        # Only one actual aembed call should have happened
        assert call_count == 1

    async def test_deduplication_propagates_error_to_waiters(self, settings):
        """If the computing coroutine fails, all waiters must receive None."""
        call_count = 0

        class FailingEmbedder(MockEmbedder):
            async def aembed(self, text: str):
                nonlocal call_count
                await asyncio.sleep(0)
                call_count += 1
                raise EmbeddingError("intentional failure")

        m = Medha(
            collection_name="dedup_fail_test",
            embedder=FailingEmbedder(dimension=384),
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()

        results = await asyncio.gather(*[
            m._get_embedding("same failing question") for _ in range(3)
        ])

        assert all(r is None for r in results)
        assert call_count == 1  # Only one actual attempt

    async def test_deduplication_unblocks_waiters_on_cancellation(self, settings):
        """If the computing coroutine is cancelled, waiters must not hang."""
        computing_started = asyncio.Event()

        class SlowEmbedder(MockEmbedder):
            async def aembed(self, text: str):
                computing_started.set()
                await asyncio.sleep(10)  # Long operation — will be cancelled
                return await super().aembed(text)

        m = Medha(
            collection_name="dedup_cancel_test",
            embedder=SlowEmbedder(dimension=384),
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()

        # Start the computing coroutine and a waiter
        compute_task = asyncio.create_task(m._get_embedding("cancel question"))
        await computing_started.wait()  # Ensure compute_task is in aembed
        waiter_task = asyncio.create_task(m._get_embedding("cancel question"))

        # Cancel the computing task
        compute_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await compute_task

        # Waiter should complete (return None or a value) — must not hang
        result = await asyncio.wait_for(waiter_task, timeout=1.0)
        assert result is None  # Waiter gets None since compute was cancelled


# ---------------------------------------------------------------------------
# Template threshold enforcement
# ---------------------------------------------------------------------------

class TestEmbeddingTimeout:
    async def test_timeout_causes_none_result(self, settings):
        """When embedding_timeout fires, _get_embedding must return None."""
        settings = Settings(
            qdrant_mode="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            embedding_timeout=0.01,  # 10 ms — will always expire
        )

        class SlowEmbedder(MockEmbedder):
            async def aembed(self, text: str):
                await asyncio.sleep(10)
                return await super().aembed(text)

        m = Medha(
            collection_name="timeout_test",
            embedder=SlowEmbedder(dimension=384),
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        result = await m._get_embedding("any question")
        assert result is None

    async def test_timeout_search_returns_error(self, settings):
        """A timed-out embedding during search must produce ERROR strategy."""
        settings = Settings(
            qdrant_mode="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            embedding_timeout=0.01,
        )

        class SlowEmbedder(MockEmbedder):
            async def aembed(self, text: str):
                await asyncio.sleep(10)
                return await super().aembed(text)

        m = Medha(
            collection_name="timeout_search_test",
            embedder=SlowEmbedder(dimension=384),
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        hit = await m.search("any question")
        assert hit.strategy == SearchStrategy.ERROR

    async def test_timeout_store_batch_returns_false(self, settings):
        """A timed-out aembed_batch must cause store_batch to return False."""
        settings = Settings(
            qdrant_mode="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            embedding_timeout=0.01,
        )

        class SlowEmbedder(MockEmbedder):
            async def aembed_batch(self, texts):
                await asyncio.sleep(10)
                return await super().aembed_batch(texts)

        m = Medha(
            collection_name="timeout_batch_test",
            embedder=SlowEmbedder(dimension=384),
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        ok = await m.store_batch([{"question": "q", "generated_query": "SELECT 1"}])
        assert ok is False


class TestBackwardCompatTemplateCollection:
    async def test_legacy_collection_triggers_warning(self, settings, caplog):
        """When a legacy '{name}_templates' collection exists, a warning is logged."""
        import logging
        backend = MockBackend()
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="myapp",
            embedder=embedder,
            backend=backend,
            settings=settings,
        )
        # Manually seed the legacy collection name with a fake entry
        from medha.types import CacheEntry
        from medha.utils.normalization import query_hash as qh
        legacy_name = "myapp_templates"
        await backend.initialize(legacy_name, 384)
        await backend.upsert(legacy_name, [
            CacheEntry(
                id="fake-id",
                vector=[0.0] * 384,
                original_question="fake",
                normalized_question="fake",
                generated_query="SELECT 1",
                query_hash=qh("SELECT 1"),
            )
        ])

        with caplog.at_level(logging.WARNING, logger="medha.core"):
            await m.start()

        assert any("Legacy template collection" in r.message for r in caplog.records)


class TestTemplateThreshold:
    async def test_low_score_template_not_returned(self, settings):
        """A template whose score is below score_threshold_template is not returned."""
        from medha.types import QueryTemplate

        # threshold is 0.70; use a template with zero keyword overlap to force low score
        no_overlap_template = QueryTemplate(
            intent="unrelated",
            template_text="zebra elephant giraffe {animal}",
            query_template="SELECT * FROM zoo WHERE animal = '{animal}'",
            parameters=["animal"],
            parameter_patterns={"animal": r"\b(cat|dog)\b"},
        )
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="threshold_test",
            embedder=embedder,
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        await m.load_templates([no_overlap_template])
        await m._l1_backend.clear()

        # This question has no keyword overlap with "zebra elephant giraffe"
        hit = await m.search("How many users are there")
        assert hit.strategy != SearchStrategy.TEMPLATE_MATCH

    async def test_high_score_template_is_returned(self, settings):
        """A template whose score meets score_threshold_template is returned."""
        from medha.types import QueryTemplate

        good_template = QueryTemplate(
            intent="count_users",
            template_text="How many users are there",
            query_template="SELECT COUNT(*) FROM users",
            parameters=[],
            priority=1,
        )
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="threshold_pass_test",
            embedder=embedder,
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        await m.load_templates([good_template])
        await m._l1_backend.clear()

        hit = await m.search("How many users are there")
        # With full keyword overlap the score should exceed 0.70
        assert hit.strategy == SearchStrategy.TEMPLATE_MATCH


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestStoreInputValidation:
    async def test_store_empty_question_returns_false(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("", "SELECT 1")
        assert ok is False

    async def test_store_whitespace_question_returns_false(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("   ", "SELECT 1")
        assert ok is False

    async def test_store_empty_query_returns_false(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("how many users", "")
        assert ok is False

    async def test_store_whitespace_query_returns_false(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("how many users", "  ")
        assert ok is False

    async def test_store_valid_inputs_succeed(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store("how many users", "SELECT COUNT(*) FROM users")
        assert ok is True


# ---------------------------------------------------------------------------
# Template sync partial failure
# ---------------------------------------------------------------------------

class TestTemplateSyncPartialFailure:
    async def test_batch_embed_failure_logs_error(self, settings, caplog):
        """If aembed_batch() raises, an ERROR is logged and no templates are synced."""
        import logging

        from medha.types import QueryTemplate

        class BatchFailEmbedder(MockEmbedder):
            async def aembed_batch(self, texts):
                raise EmbeddingError("intentional batch failure")

        template = QueryTemplate(
            intent="count_users",
            template_text="How many users",
            query_template="SELECT COUNT(*) FROM users",
            parameters=[],
            aliases=["User count please"],
        )
        backend = MockBackend()
        m = Medha(
            collection_name="batch_fail_sync_test",
            embedder=BatchFailEmbedder(dimension=384),
            backend=backend,
            settings=settings,
        )
        with caplog.at_level(logging.ERROR, logger="medha.core"):
            await m.start()
            await m.load_templates([template])

        assert any("batch embedding failed" in r.message for r in caplog.records)
        # No entries should have been synced
        assert await backend.count("__medha_templates_batch_fail_sync_test") == 0

    async def test_all_embed_failures_logs_error(self, settings, caplog):
        """If aembed_batch() always fails, an ERROR is logged and sync is aborted."""
        import logging

        from medha.types import QueryTemplate

        class AlwaysFailEmbedder(MockEmbedder):
            async def aembed_batch(self, texts):
                raise EmbeddingError("always fails")

        template = QueryTemplate(
            intent="broken_template",
            template_text="How many users",
            query_template="SELECT COUNT(*) FROM users",
            parameters=[],
        )
        backend = MockBackend()
        m = Medha(
            collection_name="all_fail_sync_test",
            embedder=AlwaysFailEmbedder(dimension=384),
            backend=backend,
            settings=settings,
        )
        with caplog.at_level(logging.ERROR, logger="medha.core"):
            await m.start()
            await m.load_templates([template])

        assert any("batch embedding failed" in r.message for r in caplog.records)
        assert await backend.count("__medha_templates_all_fail_sync_test") == 0

    async def test_batch_sync_uses_single_aembed_batch_call(self, settings):
        """_sync_templates_to_backend must call aembed_batch() exactly once."""
        from medha.types import QueryTemplate

        call_count = 0
        all_texts_received = []

        class CountingEmbedder(MockEmbedder):
            async def aembed_batch(self, texts):
                nonlocal call_count
                call_count += 1
                all_texts_received.extend(texts)
                return await super().aembed_batch(texts)

        templates = [
            QueryTemplate(
                intent="t1",
                template_text="How many users",
                query_template="SELECT COUNT(*) FROM users",
                aliases=["User count"],
            ),
            QueryTemplate(
                intent="t2",
                template_text="List products",
                query_template="SELECT * FROM products",
                aliases=["Show products", "All products"],
            ),
        ]
        m = Medha(
            collection_name="batch_count_test",
            embedder=CountingEmbedder(dimension=384),
            backend=MockBackend(),
            settings=settings,
        )
        await m.start()
        await m.load_templates(templates)

        # All 5 texts (2 template_texts + 3 aliases) in a single batch call
        assert call_count == 1
        assert len(all_texts_received) == 5
