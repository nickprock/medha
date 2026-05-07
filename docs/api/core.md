# Medha (Core)

The `Medha` class is the primary entry point. It owns the waterfall search pipeline, coordinates the vector backend, L1 cache, and embedder, and exposes all cache operations as async methods.

Construct it as an async context manager to ensure clean startup and shutdown of the backend connection and background cleanup task.

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async with Medha(
    collection="my_cache",
    embedder=FastEmbedAdapter(),
    settings=Settings(backend_type="qdrant"),
) as cache:
    await cache.store("question", "SELECT ...")
    hit = await cache.search("question")
```

---

## Medha

::: medha.core.Medha
    options:
      show_source: true
      members_order: source
