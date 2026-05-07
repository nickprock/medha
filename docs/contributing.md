# Contributing

Contributions are welcome — bug fixes, new backends, new embedders, documentation improvements, and test coverage expansions.

---

## Development Setup

Clone the repo and install all extras with `uv`:

```bash
git clone https://github.com/ArchAI-Labs/medha.git
cd medha
uv sync --all-extras
```

Install pre-commit hooks:

```bash
pre-commit install
```

The hooks run `ruff` (lint + format), `mypy` (type check), and `pytest` (fast unit tests) on every commit.

---

## Running Tests

Tests are organized with pytest markers:

```bash
# Fast unit tests only (no external services)
pytest -m unit

# Integration tests against Qdrant (requires Docker)
docker run -d -p 6333:6333 qdrant/qdrant
pytest -m integration

# All tests
pytest

# With coverage report
pytest --cov=medha --cov-report=term-missing
```

Marker definitions are in `pyproject.toml` under `[tool.pytest.ini_options]`.

---

## Adding a New Backend

1. Create `src/medha/backends/my_backend.py`
2. Implement all abstract methods from `VectorStorageBackend`:

    ```python
    from medha.interfaces.storage import VectorStorageBackend
    from medha.types import CacheEntry

    class MyBackend(VectorStorageBackend):
        async def initialize(self, collection: str, dimension: int) -> None:
            # Connect and create/verify the collection
            ...

        async def upsert(self, entries: list[CacheEntry]) -> None:
            # Insert or update entries
            ...

        async def query(
            self, vector: list[float], top_k: int
        ) -> list[tuple[CacheEntry, float]]:
            # Return (entry, cosine_score) pairs sorted by score descending
            ...

        async def delete(self, entry_ids: list[str]) -> None:
            ...

        async def count(self) -> int:
            ...

        async def close(self) -> None:
            # Release connections / cleanup resources
            ...
    ```

3. Register the backend in `src/medha/backends/__init__.py`:

    ```python
    BACKEND_REGISTRY = {
        ...
        "my_backend": "medha.backends.my_backend.MyBackend",
    }
    ```

4. Add the optional dependency in `pyproject.toml`:

    ```toml
    [project.optional-dependencies]
    my_backend = ["my-backend-client>=1.0"]
    ```

5. Add `!!! info "Install"` and a config snippet to [backends.md](user_guide/backends.md)
6. Write integration tests in `tests/backends/test_my_backend.py` with `@pytest.mark.integration`

---

## Adding a New Embedder

1. Create `src/medha/embeddings/my_embedder_adapter.py`
2. Subclass `BaseEmbedder`:

    ```python
    from medha.interfaces.embedder import BaseEmbedder

    class MyEmbedderAdapter(BaseEmbedder):
        def __init__(self, api_key: str, model: str = "default-model"):
            self._client = MyEmbedderClient(api_key=api_key)
            self._model = model

        async def embed(self, text: str) -> list[float]:
            response = await self._client.embed(text, model=self._model)
            return response.vector

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            responses = await self._client.embed_batch(texts, model=self._model)
            return [r.vector for r in responses]
    ```

3. Add the optional dependency to `pyproject.toml`
4. Document it in [embedders.md](user_guide/embedders.md)
5. Write unit tests in `tests/embeddings/test_my_embedder_adapter.py`

---

## Code Style

| Rule | Value |
|---|---|
| Linter | `ruff` (configured in `pyproject.toml`) |
| Type checker | `mypy --strict` |
| Line length | 120 characters |
| Docstrings | Google style |
| Import order | `ruff` isort-compatible |

Run checks locally:

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

---

## Submitting a Pull Request

1. **Branch** from `main`: `git checkout -b feat/my-feature`
2. **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/):
    - `feat: add LanceDB backend`
    - `fix: handle Qdrant timeout on large collections`
    - `docs: add pgvector configuration example`
    - `test: add integration tests for Chroma backend`
3. **Changelog** — add an entry to `CHANGELOG.md` under `[Unreleased]`
4. **Open the PR** against `main` with a description that explains *why*, not just *what*
5. All CI checks must pass before merge: `ruff`, `mypy`, unit tests, and `mkdocs build --strict`
