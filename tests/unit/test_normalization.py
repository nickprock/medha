"""Unit tests for medha.utils.normalization."""

from medha.utils.normalization import normalize_question, question_hash, query_hash


class TestNormalizeQuestion:
    def test_basic_normalization(self):
        result = normalize_question("Show me the top 5 products!")
        assert result == "show top 5 products"

    def test_empty_input(self):
        assert normalize_question("") == ""

    def test_whitespace_only(self):
        assert normalize_question("   ") == ""

    def test_whitespace_collapse(self):
        result = normalize_question("show   me   the    products")
        assert "  " not in result

    def test_extra_replacements(self):
        extras = [(r"\bfoo\b", "bar")]
        result = normalize_question("foo products", extra_replacements=extras)
        assert "bar" in result
        assert "foo" not in result

    def test_trailing_punctuation_removed(self):
        result = normalize_question("how many users?")
        assert not result.endswith("?")

    def test_please_removed(self):
        result = normalize_question("please show me users")
        assert "please" not in result


class TestQuestionHash:
    def test_deterministic(self):
        h1 = question_hash("How many users?")
        h2 = question_hash("How many users?")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = question_hash("FOO")
        h2 = question_hash("foo")
        assert h1 == h2

    def test_returns_32_char_hex(self):
        h = question_hash("test")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


class TestQueryHash:
    def test_query_hash(self):
        h = query_hash("SELECT 1")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_query_hash_strips_whitespace(self):
        h1 = query_hash("SELECT 1")
        h2 = query_hash("  SELECT 1  ")
        assert h1 == h2
