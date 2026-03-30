# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-03-30

### Added

- **`InMemoryBackend`**: pure-Python in-process vector backend with zero external
  dependencies. Useful for testing, development, CI, and lightweight deployments.
  Select via `Settings(backend_type="memory")` or env var `MEDHA_BACKEND_TYPE=memory`.

- **`PgVectorBackend`**: PostgreSQL + pgvector vector backend for production deployments
  that already run PostgreSQL. Install with `pip install medha-archai[pgvector]`.
  Select via `Settings(backend_type="pgvector")` or env var `MEDHA_BACKEND_TYPE=pgvector`.

- **`Settings.backend_type`**: new `Literal["qdrant", "memory", "pgvector"]` field to
  select the vector backend declaratively without passing an instance to `Medha`.

- **PostgreSQL configuration fields** in `Settings`: `pg_dsn`, `pg_host`, `pg_port`,
  `pg_database`, `pg_user`, `pg_password` (SecretStr), `pg_schema`, `pg_table_prefix`,
  `pg_pool_min_size`, `pg_pool_max_size`.

- **Backend factory** in `Medha.__init__`: `backend=None` now resolves to the correct
  backend class based on `settings.backend_type` instead of always defaulting to Qdrant.

- **`connect()` method** added to `VectorStorageBackend` base class as a concrete no-op,
  ensuring all backends share the same lifecycle interface.

- **`Settings.max_question_length`** (default `8192`): rejects oversized question strings
  in `search()` (returns `SearchStrategy.ERROR`) and `store()` (raises `ValueError`),
  preventing DoS via oversized inputs.

- **`Settings.max_file_size_mb`** (default `100`): `warm_from_file()` and
  `load_templates_from_file()` reject files exceeding this limit before reading them.

- **`Settings.allowed_file_dir`** (default `None`): when set, restricts
  `warm_from_file()` and `load_templates_from_file()` to paths inside the specified
  directory, preventing path traversal attacks.

- **`Medha._resolve_and_check_path()`**: private method implementing path resolution
  and `allowed_file_dir` enforcement, used by all file-loading methods.

- **`pg_schema` and `pg_table_prefix` validators**: identifier regex
  `^[a-zA-Z_][a-zA-Z0-9_]{0,62}$` enforced at `Settings` construction time,
  preventing SQL injection via misconfigured table names.

- New demo notebooks:
  - `demo/11_inmemory_backend.ipynb` — zero-dependency setup, testing patterns,
    performance characteristics of the linear scan backend
  - `demo/12_pgvector_backend.ipynb` — PostgreSQL + pgvector setup, table naming,
    pool tuning, multi-tenant isolation, persistence verification

- Updated existing demo notebooks (`01`–`10`) to use explicit `backend_type` in
  all `Settings(...)` calls and to demonstrate new security settings where relevant.

### Changed

- **`Settings.qdrant_api_key`**: type changed from `str | None` to `SecretStr | None`.
  The secret value is never exposed in logs or `repr()`. Callers passing a plain
  string continue to work — Pydantic coerces it automatically.

- **`QdrantBackend._build_client()`**: updated to call `.get_secret_value()` when
  constructing the Qdrant Cloud client from `qdrant_api_key`.

- **`Medha.__init__`**: when `backend=None`, the backend is now selected via
  `settings.backend_type` (default `"qdrant"` — **fully backward compatible**).

- **`pyproject.toml`**: development status promoted from Alpha (3) to Beta (4).
  Added `[pgvector]` optional dependency group. `[all]` updated to include it.

- **Experiments** (`experiments/`): all benchmark scripts updated to use
  `backend_type="memory"` (zero-infra) where the backend is not the subject of
  measurement. `latency_benchmark.py` gains a `--backend qdrant|memory` flag for
  direct comparison.

### Fixed

- `qdrant_api_key` was previously logged as a plain string in debug output.
  Now stored as `SecretStr` and masked automatically.

- `warm_from_file()` and `load_templates_from_file()` previously allowed arbitrary
  file paths and sizes. Now validated against `allowed_file_dir` and `max_file_size_mb`.

- `pg_schema` and `pg_table_prefix` were previously passed directly to SQL queries
  without validation. Now validated at config construction time.

### Known Issues

- **`qdrant-client` is still a core dependency** even when `backend_type="memory"` or
  `backend_type="pgvector"` is selected. Users who only need `InMemoryBackend` or
  `PgVectorBackend` will still have `qdrant-client` installed unnecessarily.
  Moving it to an optional `[qdrant]` extra would be a breaking change (it would
  require existing users to add `[qdrant]` to their install command) and is deferred
  to 0.3.0, where it can be versioned as a breaking change alongside a potential
  change to the `backend_type` default.

## [0.1.0] — 2025-02

Initial release.

- Waterfall search strategy (L1 Cache → Template → Exact → Semantic → Fuzzy)
- Qdrant backend (memory / Docker / Cloud modes)
- FastEmbed and OpenAI embedding adapters
- In-memory and Redis L1 cache
- Template matching with parameter extraction (regex, spaCy, GLiNER, heuristics)
- Pydantic-based configuration with `MEDHA_` env var prefix
- Persistent embedding cache (JSON file)
- Comprehensive unit and integration test suite
- 10 demo notebooks
