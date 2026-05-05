"""Unit tests for medha.config.Settings."""

import pytest
from pydantic import ValidationError

from medha.config import Settings


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.qdrant_mode == "memory"
        assert s.score_threshold_exact == 0.99
        assert s.score_threshold_semantic == 0.85
        assert s.l1_cache_max_size == 1000

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            Settings(qdrant_mode="invalid")

    def test_exact_threshold_too_low(self):
        with pytest.raises(ValidationError):
            Settings(score_threshold_exact=0.5)

    def test_semantic_must_be_below_exact(self):
        with pytest.raises(ValidationError):
            Settings(score_threshold_exact=0.99, score_threshold_semantic=0.99)

    def test_env_var_loading(self, monkeypatch):
        monkeypatch.setenv("MEDHA_QDRANT_MODE", "docker")
        s = Settings()
        assert s.qdrant_mode == "docker"

    def test_settings_dump(self):
        s = Settings()
        d = s.model_dump()
        assert isinstance(d, dict)
        assert "qdrant_mode" in d
        assert "score_threshold_exact" in d

    def test_fuzzy_prefilter_defaults(self):
        s = Settings()
        assert s.score_threshold_fuzzy_prefilter == 0.65
        assert s.fuzzy_prefilter_top_k == 50

    def test_fuzzy_prefilter_threshold_bounds(self):
        with pytest.raises(ValidationError):
            Settings(score_threshold_fuzzy_prefilter=1.5)
        with pytest.raises(ValidationError):
            Settings(score_threshold_fuzzy_prefilter=-0.1)

    def test_fuzzy_prefilter_top_k_bounds(self):
        with pytest.raises(ValidationError):
            Settings(fuzzy_prefilter_top_k=0)
        with pytest.raises(ValidationError):
            Settings(fuzzy_prefilter_top_k=1001)

    def test_fuzzy_prefilter_custom_values(self):
        s = Settings(score_threshold_fuzzy_prefilter=0.50, fuzzy_prefilter_top_k=100)
        assert s.score_threshold_fuzzy_prefilter == 0.50
        assert s.fuzzy_prefilter_top_k == 100


class TestBackendTypeSettings:
    def test_backend_type_default(self):
        s = Settings()
        assert s.backend_type == "memory"

    def test_backend_type_memory(self):
        s = Settings(backend_type="memory")
        assert s.backend_type == "memory"

    def test_backend_type_pgvector(self):
        s = Settings(backend_type="pgvector")
        assert s.backend_type == "pgvector"

    def test_backend_type_invalid(self):
        with pytest.raises(ValidationError):
            Settings(backend_type="unknown")

    def test_pg_pool_max_lt_min_raises(self):
        with pytest.raises(ValidationError):
            Settings(backend_type="pgvector", pg_pool_min_size=5, pg_pool_max_size=2)

    def test_pg_dsn_overrides_fields(self):
        # DSN è una stringa, non viene parsata — solo passata ad asyncpg
        s = Settings(pg_dsn="postgresql://user:pass@host/db")
        assert s.pg_dsn == "postgresql://user:pass@host/db"

    def test_backend_type_vectorchord(self):
        s = Settings(backend_type="vectorchord")
        assert s.backend_type == "vectorchord"

    def test_backend_type_elasticsearch(self):
        s = Settings(backend_type="elasticsearch")
        assert s.backend_type == "elasticsearch"

    def test_backend_type_chroma(self):
        s = Settings(backend_type="chroma")
        assert s.backend_type == "chroma"

    def test_backend_type_weaviate(self):
        s = Settings(backend_type="weaviate")
        assert s.backend_type == "weaviate"

    def test_backend_type_redis(self):
        s = Settings(backend_type="redis")
        assert s.backend_type == "redis"

    def test_backend_type_azure_search(self):
        s = Settings(backend_type="azure-search")
        assert s.backend_type == "azure-search"


class TestElasticsearchSettings:
    def test_defaults(self):
        s = Settings(backend_type="elasticsearch")
        assert s.es_hosts == ["http://localhost:9200"]
        assert s.es_index_prefix == "medha"
        assert s.es_num_candidates == 100
        assert s.es_timeout == 30.0

    def test_custom_hosts(self):
        s = Settings(es_hosts=["http://es1:9200", "http://es2:9200"])
        assert len(s.es_hosts) == 2

    def test_api_key_stored_as_secret(self):
        s = Settings(es_api_key="mykey")
        assert s.es_api_key is not None
        assert s.es_api_key.get_secret_value() == "mykey"


class TestChromaSettings:
    def test_defaults(self):
        s = Settings(backend_type="chroma")
        assert s.chroma_mode == "ephemeral"
        assert s.chroma_host == "localhost"
        assert s.chroma_port == 8000

    def test_http_mode(self):
        s = Settings(chroma_mode="http", chroma_host="my-chroma.example.com", chroma_port=8080)
        assert s.chroma_mode == "http"
        assert s.chroma_host == "my-chroma.example.com"

    def test_invalid_mode(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(chroma_mode="invalid_mode")

    def test_persist_path(self):
        s = Settings(chroma_persist_path="/data/chroma")
        assert s.chroma_persist_path == "/data/chroma"


class TestWeaviateSettings:
    def test_defaults(self):
        s = Settings(backend_type="weaviate")
        assert s.weaviate_mode == "local"
        assert s.weaviate_host == "localhost"
        assert s.weaviate_http_port == 8080
        assert s.weaviate_grpc_port == 50051
        assert s.weaviate_collection_prefix == "Medha"

    def test_cloud_mode(self):
        s = Settings(weaviate_mode="cloud", weaviate_cloud_url="https://my-cluster.weaviate.network")
        assert s.weaviate_mode == "cloud"
        assert s.weaviate_cloud_url == "https://my-cluster.weaviate.network"

    def test_api_key_stored_as_secret(self):
        s = Settings(weaviate_api_key="wv-key")
        assert s.weaviate_api_key is not None
        assert s.weaviate_api_key.get_secret_value() == "wv-key"


class TestRedisSettings:
    def test_defaults(self):
        s = Settings(backend_type="redis")
        assert s.redis_mode == "standalone"
        assert s.redis_host == "localhost"
        assert s.redis_port == 6379
        assert s.redis_db == 0
        assert s.redis_key_prefix == "medha"
        assert s.redis_index_algorithm == "HNSW"

    def test_url_setting(self):
        s = Settings(redis_url="redis://myhost:6380/1")
        assert s.redis_url == "redis://myhost:6380/1"

    def test_hnsw_params(self):
        s = Settings(redis_hnsw_m=32, redis_hnsw_ef_construction=400)
        assert s.redis_hnsw_m == 32
        assert s.redis_hnsw_ef_construction == 400

    def test_flat_algorithm(self):
        s = Settings(redis_index_algorithm="FLAT")
        assert s.redis_index_algorithm == "FLAT"

    def test_invalid_algorithm(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(redis_index_algorithm="INVALID")

    def test_sentinel_mode(self):
        s = Settings(redis_mode="sentinel", redis_sentinel_master="master1")
        assert s.redis_mode == "sentinel"
        assert s.redis_sentinel_master == "master1"


class TestAzureSearchSettings:
    def test_defaults(self):
        s = Settings(backend_type="azure-search")
        assert s.azure_search_endpoint == ""
        assert s.azure_search_index_name == "medha"
        assert s.azure_search_api_version == "2024-05-01-preview"
        assert s.azure_search_top_k_candidates == 50

    def test_api_key_stored_as_secret(self):
        s = Settings(azure_search_api_key="az-key")
        assert s.azure_search_api_key is not None
        assert s.azure_search_api_key.get_secret_value() == "az-key"

    def test_custom_endpoint(self):
        s = Settings(azure_search_endpoint="https://svc.search.windows.net")
        assert s.azure_search_endpoint == "https://svc.search.windows.net"

    def test_top_k_candidates_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(azure_search_top_k_candidates=0)
        with pytest.raises(ValidationError):
            Settings(azure_search_top_k_candidates=10001)


class TestVectorChordSettings:
    def test_defaults(self):
        s = Settings(backend_type="vectorchord")
        assert s.vc_lists == [1000]
        assert s.vc_residual_quantization is True

    def test_custom_lists(self):
        s = Settings(vc_lists=[500, 1000])
        assert s.vc_lists == [500, 1000]

    def test_residual_disabled(self):
        s = Settings(vc_residual_quantization=False)
        assert s.vc_residual_quantization is False


class TestCacheLifecycleSettings:
    def test_default_ttl_none(self):
        s = Settings()
        assert s.default_ttl_seconds is None

    def test_custom_ttl(self):
        s = Settings(default_ttl_seconds=3600)
        assert s.default_ttl_seconds == 3600

    def test_cleanup_interval_none(self):
        s = Settings()
        assert s.cleanup_interval_seconds is None

    def test_cleanup_interval_minimum(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(cleanup_interval_seconds=30)  # must be >= 60

    def test_cleanup_interval_valid(self):
        s = Settings(cleanup_interval_seconds=300)
        assert s.cleanup_interval_seconds == 300
