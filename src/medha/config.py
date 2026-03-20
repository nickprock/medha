"""Pydantic-based configuration with environment variable support (MEDHA_ prefix)."""

from typing import Literal, Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for a Medha instance."""

    model_config = SettingsConfigDict(
        env_prefix="MEDHA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Backend ---
    qdrant_mode: Literal["memory", "docker", "cloud"] = Field(
        default="memory",
        description="Qdrant connection mode",
    )
    qdrant_host: str = Field(default="localhost", description="Qdrant host for docker/cloud mode")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Qdrant gRPC port")
    qdrant_url: Optional[str] = Field(default=None, description="Full Qdrant URL (overrides host:port)")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant Cloud API key")

    # --- Query language ---
    query_language: Literal["sql", "cypher", "graphql", "generic"] = Field(
        default="generic",
        description="Target query language (informational, no behavioral change)",
    )

    # --- Search thresholds ---
    score_threshold_exact: float = Field(default=0.99, ge=0.0, le=1.0)
    score_threshold_semantic: float = Field(default=0.90, ge=0.0, le=1.0)
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

    # --- Quantization search ---
    quantization_rescore: bool = Field(
        default=True,
        description="Re-score top results using original vectors after quantized search",
    )
    quantization_oversampling: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="Oversampling factor for quantized search (e.g. 2.0 fetches 2x candidates before re-scoring). None = Qdrant default",
    )
    quantization_ignore: bool = Field(
        default=False,
        description="Bypass quantized vectors and search only original vectors",
    )
    quantization_always_ram: bool = Field(
        default=True,
        description="Keep quantized vectors in RAM. Combined with on_disk=True enables hybrid storage (original on disk, quantized in RAM)",
    )

    # --- Template loading ---
    template_file: Optional[str] = Field(default=None, description="Path to JSON template file")

    # --- Persistent embedding cache ---
    embedding_cache_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a JSON file used to persist the embedding cache across restarts. "
            "When set, embeddings are loaded from disk on start() and saved to disk on close(). "
            "Speeds up warm-start scenarios where the same questions recur across sessions."
        ),
    )

    # --- Batch operations ---
    batch_size: int = Field(default=100, ge=1, le=10000, description="Batch size for bulk upsert")

    # --- Timeouts ---
    embedding_timeout: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Timeout in seconds for embedding calls (aembed and aembed_batch). "
            "None disables the timeout. Increase for large batches or slow networks."
        ),
    )

    # --- Validators ---
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
