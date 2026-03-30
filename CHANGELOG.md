# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-04-XX

### Added

- **`InMemoryBackend`**: pure Python vector backend with zero external dependencies.
  Useful for testing, development, and lightweight deployments.
  Select via `Settings(backend_type="memory")` or env var `MEDHA_BACKEND_TYPE=memory`.

- **`PgVectorBackend`**: PostgreSQL + pgvector vector backend for production deployments
  that already run PostgreSQL. Install with `pip install medha-archai[pgvector]`.
  Select via `Settings(backend_type="pgvector")` or env var `MEDHA_BACKEND_TYPE=pgvector`.

- **`Settings.backend_type`**: new field to select the vector backend without
  passing a backend instance explicitly to `Medha`.

- **PostgreSQL configuration fields** in `Settings`: `pg_dsn`, `pg_host`, `pg_port`,
  `pg_database`, `pg_user`, `pg_password`, `pg_schema`, `pg_table_prefix`,
  `pg_pool_min_size`, `pg_pool_max_size`.

- **Backend factory** in `Medha.__init__`: `backend=None` now resolves to the correct
  backend class based on `settings.backend_type` instead of always defaulting to Qdrant.

- **`connect()` abstract method** added to `VectorStorageBackend` base class as a
  concrete no-op, ensuring all backends share the same lifecycle interface.

- New demo notebooks:
  - `demo/11_inmemory_backend.ipynb`: zero-dependency setup with InMemoryBackend
  - `demo/12_pgvector_backend.ipynb`: PostgreSQL + pgvector setup

### Changed

- `Medha.__init__`: when `backend=None`, the backend is now selected via
  `settings.backend_type` (default `"qdrant"` — **fully backward compatible**).

- `pyproject.toml`: development status promoted from Alpha (3) to Beta (4).

### Fixed

- Nothing.

## [0.1.0] — 2025-XX-XX

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
