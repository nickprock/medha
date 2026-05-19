# Security

Medha includes built-in defenses for deployments where questions and file paths come from untrusted input. Three `Settings` fields cover the most common attack vectors.

---

## Input Length Guard — `max_question_length`

Prevents DoS via oversized question strings. When the limit is exceeded:

- `search()` returns `SearchStrategy.ERROR` without raising
- `store()` raises `ValueError`

```python
from medha import Settings

settings = Settings(max_question_length=2048)  # default: 8192
```

```python
async with Medha("my_cache", embedder=embedder, settings=settings) as cache:
    hit = await cache.search("A" * 10_000)
    print(hit.strategy)  # SearchStrategy.ERROR
```

---

## File Size Limit — `max_file_size_mb`

`warm_from_file()` and `load_templates_from_file()` reject files larger than this limit **before reading them** — preventing memory exhaustion from malicious uploads.

```python
settings = Settings(max_file_size_mb=50)  # default: 100 MB
```

---

## Path Traversal Protection — `allowed_file_dir`

When set, `warm_from_file()` and `load_templates_from_file()` reject any path that resolves outside the specified directory. This blocks `../` traversal attacks when file paths come from user input.

```python
settings = Settings(allowed_file_dir="/app/data")
```

```python
# Raises ValueError — path resolves outside /app/data
await cache.warm_from_file("/app/data/../etc/passwd")
```

---

## Secret Handling — `qdrant_api_key`

`qdrant_api_key` is stored as Pydantic `SecretStr` — it is **never logged or printed**, even in debug mode.

```python
settings = Settings(
    backend_type="qdrant",
    qdrant_mode="cloud",
    qdrant_url="https://your-cluster.cloud.qdrant.io",
    qdrant_api_key="your-secret-key",  # stored as SecretStr, never exposed
)
```

Prefer environment variables over hardcoded values:

```bash
export MEDHA_QDRANT_API_KEY=your-secret-key
```

---

## PostgreSQL Identifier Validation

When using `pgvector` or `vectorchord`, Medha validates collection names and column identifiers against a strict allowlist (`[a-zA-Z0-9_]`, max 63 chars) before interpolating them into SQL. This prevents SQL injection via crafted collection names.

```python
# Raises ValueError — invalid identifier
settings = Settings(backend_type="pgvector", pg_dsn="postgresql://...")
cache = Medha("my_cache; DROP TABLE users;--", embedder=embedder, settings=settings)
await cache.start()  # ValueError: invalid collection name
```

---

## Recommended Production Configuration

```python
from medha import Settings

settings = Settings(
    # Input guard
    max_question_length=4096,

    # File operation guards
    max_file_size_mb=50,
    allowed_file_dir="/app/data",

    # Credentials via env vars (MEDHA_QDRANT_API_KEY)
    backend_type="qdrant",
    qdrant_mode="cloud",
    qdrant_url="https://your-cluster.cloud.qdrant.io",
)
```

```bash
# .env
MEDHA_QDRANT_API_KEY=your-secret-key
MEDHA_MAX_QUESTION_LENGTH=4096
MEDHA_MAX_FILE_SIZE_MB=50
MEDHA_ALLOWED_FILE_DIR=/app/data
```
