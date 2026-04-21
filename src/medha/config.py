"""Pydantic-based configuration with environment variable support (MEDHA_ prefix)."""

import re
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


class Settings(BaseSettings):
    """Central configuration for a Medha instance."""

    model_config = SettingsConfigDict(
        env_prefix="MEDHA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Backend selection ---
    backend_type: Literal["qdrant", "memory", "pgvector", "elasticsearch", "vectorchord"] = Field(
        default="memory",
        description=(
            "Vector storage backend to use. "
            "'memory' uses pure Python in-process storage, zero external deps (default). "
            "'qdrant' requires qdrant-client (pip install medha-archai[qdrant]). "
            "'pgvector' requires asyncpg and pgvector (pip install medha-archai[pgvector]). "
            "'elasticsearch' requires elasticsearch[async]>=8.12 (pip install medha-archai[elasticsearch]). "
            "'vectorchord' requires asyncpg (pip install medha-archai[vectorchord])."
        ),
    )

    # --- Backend ---
    qdrant_mode: Literal["memory", "docker", "cloud"] = Field(
        default="memory",
        description="Qdrant connection mode",
    )
    qdrant_host: str = Field(default="localhost", description="Qdrant host for docker/cloud mode")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Qdrant gRPC port")
    qdrant_url: str | None = Field(default=None, description="Full Qdrant URL (overrides host:port)")
    qdrant_api_key: SecretStr | None = Field(default=None, description="Qdrant Cloud API key")

    # --- Query language ---
    query_language: Literal["sql", "cypher", "graphql", "generic"] = Field(
        default="generic",
        description="Target query language (informational, no behavioral change)",
    )

    # --- Search thresholds ---
    score_threshold_exact: float = Field(default=0.99, ge=0.0, le=1.0)
    score_threshold_semantic: float = Field(default=0.85, ge=0.0, le=1.0)
    score_threshold_template: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum score for a template match to be returned. "
            "The maximum achievable score is ~0.88 "
            "(keyword_bonus=1.0 × 0.5 + param_completeness=1.0 × 0.3 + priority_1 × 0.08). "
            "Values above 0.88 will never match any template."
        ),
    )
    score_threshold_fuzzy: float = Field(default=85.0, ge=0.0, le=100.0, description="Fuzzy match threshold (0-100)")
    score_threshold_fuzzy_prefilter: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for the vector pre-filter used in fuzzy search. "
            "Candidates below this threshold are excluded before Levenshtein scoring, "
            "reducing fuzzy search from O(n) to O(top_k). Lower values increase recall "
            "at the cost of more fuzzy comparisons."
        ),
    )
    fuzzy_prefilter_top_k: int = Field(
        default=50,
        ge=1,
        le=1000,
        description=(
            "Maximum number of vector-similar candidates to retrieve for fuzzy pre-filtering. "
            "Fuzzy scoring is applied only to these candidates instead of the full collection."
        ),
    )

    # --- L1 Cache ---
    l1_cache_max_size: int = Field(default=1000, ge=0, description="Max entries in L1 in-memory cache (0=disabled)")

    # --- Qdrant tuning ---
    hnsw_m: int = Field(default=16, ge=4, le=64, description="HNSW edges per node")
    hnsw_ef_construct: int = Field(default=100, ge=50, le=500, description="HNSW construction search depth")
    enable_quantization: bool = Field(default=True, description="Enable vector quantization")
    quantization_type: Literal["scalar", "binary"] = Field(
        default="scalar",
        description="Quantization method. Binary only for dim >= 512",
    )
    on_disk: bool = Field(default=False, description="Store vectors on disk (large datasets)")

    # --- PostgreSQL / pgvector ---
    pg_dsn: str | None = Field(
        default=None,
        description=(
            "Full asyncpg DSN for PostgreSQL connection "
            "(e.g. 'postgresql://user:pass@localhost:5432/dbname'). "
            "When set, overrides pg_host, pg_port, pg_database, pg_user, pg_password."
        ),
    )
    pg_host: str = Field(default="localhost", description="PostgreSQL host")
    pg_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    pg_database: str = Field(default="medha", description="PostgreSQL database name")
    pg_user: str = Field(default="postgres", description="PostgreSQL user")
    pg_password: SecretStr = Field(default=SecretStr(""), description="PostgreSQL password")
    pg_schema: str = Field(default="public", description="PostgreSQL schema for Medha tables")
    pg_table_prefix: str = Field(
        default="medha",
        description="Prefix for Medha table names (e.g. 'medha' → table 'medha_my_cache')",
    )
    pg_pool_min_size: int = Field(default=2, ge=1, description="Min connections in asyncpg pool")
    pg_pool_max_size: int = Field(default=10, ge=1, description="Max connections in asyncpg pool")

    # --- VectorChord ---
    vc_lists: list[int] = Field(
        default_factory=lambda: [1000],
        description="Number of centroids per level for the vchordrq index.",
    )
    vc_residual_quantization: bool = Field(
        default=True,
        description="Enable residual quantization in the vchordrq index.",
    )

    # --- Elasticsearch ---
    es_hosts: list[str] = Field(
        default_factory=lambda: ["http://localhost:9200"],
        description="Elasticsearch node URLs",
    )
    es_api_key: SecretStr | None = Field(default=None, description="Elasticsearch API key")
    es_username: str | None = Field(default=None, description="Elasticsearch basic-auth username")
    es_password: SecretStr | None = Field(default=None, description="Elasticsearch basic-auth password")
    es_index_prefix: str = Field(default="medha", description="Prefix for Elasticsearch index names")
    es_num_candidates: int = Field(
        default=100, ge=1, le=10000, description="num_candidates for kNN search"
    )
    es_timeout: float = Field(default=30.0, gt=0.0, description="Request timeout in seconds")

    # --- Quantization search ---
    quantization_rescore: bool = Field(
        default=True,
        description="Re-score top results using original vectors after quantized search",
    )
    quantization_oversampling: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Oversampling factor for quantized search "
            "(e.g. 2.0 fetches 2x candidates before re-scoring). None = Qdrant default"
        ),
    )
    quantization_ignore: bool = Field(
        default=False,
        description="Bypass quantized vectors and search only original vectors",
    )
    quantization_always_ram: bool = Field(
        default=True,
        description=(
            "Keep quantized vectors in RAM. Combined with on_disk=True enables "
            "hybrid storage (original on disk, quantized in RAM)"
        ),
    )

    # --- Template loading ---
    template_file: str | None = Field(default=None, description="Path to JSON template file")

    # --- Persistent embedding cache ---
    embedding_cache_path: str | None = Field(
        default=None,
        description=(
            "Path to a JSON file used to persist the embedding cache across restarts. "
            "When set, embeddings are loaded from disk on start() and saved to disk on close(). "
            "Speeds up warm-start scenarios where the same questions recur across sessions."
        ),
    )

    # --- File operations ---
    allowed_file_dir: str | None = Field(
        default=None,
        description=(
            "If set, warm_from_file() and load_templates_from_file() will reject paths "
            "outside this directory. Useful when the caller is not trusted. "
            "Example: '/app/data'. Default: None (no restriction)."
        ),
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description=(
            "Maximum file size in MB for warm_from_file() and load_templates_from_file(). "
            "Files exceeding this limit are rejected before reading."
        ),
    )

    # --- Input validation ---
    max_question_length: int = Field(
        default=8192,
        ge=64,
        le=1_000_000,
        description=(
            "Maximum allowed length (characters) for a question string. "
            "Questions exceeding this limit are rejected with SearchStrategy.ERROR "
            "to prevent DoS via oversized inputs. Default: 8192 chars (~8KB)."
        ),
    )

    # --- Cache lifecycle ---
    default_ttl_seconds: int | None = Field(
        default=None,
        ge=1,
        description=(
            "TTL di default in secondi per le nuove entry. "
            "None = entry immortali (comportamento attuale). "
            "Può essere sovrascritto entry per entry tramite il parametro ttl di store()."
        ),
    )
    cleanup_interval_seconds: int | None = Field(
        default=None,
        ge=60,
        description=(
            "Intervallo in secondi per il cleanup automatico delle entry scadute. "
            "None = nessun cleanup automatico. "
            "Se impostato, Medha.start() avvia un task asyncio che chiama expire() periodicamente."
        ),
    )

    # --- Batch operations ---
    batch_size: int = Field(default=100, ge=1, le=10000, description="Batch size for bulk upsert")

    # --- Timeouts ---
    embedding_timeout: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Timeout in seconds for embedding calls (aembed and aembed_batch). "
            "None disables the timeout. Increase for large batches or slow networks."
        ),
    )

    # --- Validators ---
    @field_validator("pg_schema", "pg_table_prefix")
    @classmethod
    def validate_pg_identifier(cls, v: str) -> str:
        if not _SAFE_IDENTIFIER_RE.match(v):
            raise ValueError(
                f"Invalid PostgreSQL identifier '{v}': must match ^[a-zA-Z_][a-zA-Z0-9_]{{0,62}}$"
            )
        return v

    @field_validator("pg_pool_max_size")
    @classmethod
    def pool_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        min_size = info.data.get("pg_pool_min_size", 2)
        if v < min_size:
            raise ValueError(
                f"pg_pool_max_size ({v}) must be >= pg_pool_min_size ({min_size})"
            )
        return v

    @field_validator("score_threshold_exact")
    @classmethod
    def exact_must_be_high(cls, v: float) -> float:
        if v < 0.90:
            raise ValueError("Exact threshold should be >= 0.90 to avoid false positives")
        return v

    @field_validator("score_threshold_semantic")
    @classmethod
    def semantic_below_exact(cls, v: float, info: ValidationInfo) -> float:
        exact = info.data.get("score_threshold_exact", 0.99)
        if v >= exact:
            raise ValueError("Semantic threshold must be lower than exact threshold")
        return v
