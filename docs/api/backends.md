# Backends

Backend classes implement the `VectorStorageBackend` ABC. You generally do not instantiate them directly — `Medha` creates the correct backend based on `Settings.backend_type`. These docs are provided for contributors and advanced users who need to extend or inspect backend behaviour.

See the [Backends guide](../user_guide/backends.md) for installation instructions and configuration examples.

---

## InMemoryBackend

::: medha.backends.memory.InMemoryBackend

---

## QdrantBackend

!!! note "Conditional Import"

    `QdrantBackend` is only importable when the `qdrant` extra is installed (`pip install "medha-archai[qdrant]"`). Importing it without the extra raises `ImportError` with a helpful install hint.

::: medha.backends.qdrant.QdrantBackend

---

## PgVectorBackend

::: medha.backends.pgvector.PgVectorBackend

---

## ElasticsearchBackend

::: medha.backends.elasticsearch.ElasticsearchBackend

---

## VectorChordBackend

::: medha.backends.vectorchord.VectorChordBackend

---

## ChromaBackend

::: medha.backends.chroma.ChromaBackend

---

## WeaviateBackend

::: medha.backends.weaviate.WeaviateBackend

---

## RedisVectorBackend

::: medha.backends.redis_vector.RedisVectorBackend

---

## AzureSearchBackend

::: medha.backends.azure_search.AzureSearchBackend

---

## LanceDBBackend

::: medha.backends.lancedb.LanceDBBackend
