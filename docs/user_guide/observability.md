# Observability

Medha exposes hit/miss counters, per-strategy breakdowns, and latency percentiles through the `CacheStats` object and standard Python logging.

---

## `CacheStats` Fields

Retrieve statistics with `get_stats()`:

```python
async with Medha("demo", embedder=embedder, settings=settings) as cache:
    # ... store and search calls ...
    stats = await cache.get_stats()
```

| Field | Type | Description |
|---|---|---|
| `total_hits` | `int` | Number of successful cache lookups |
| `total_misses` | `int` | Number of cache misses |
| `hit_rate` | `float` | Fraction of requests that hit the cache |
| `avg_latency_ms` | `float` | Mean search latency across all requests |
| `p50_latency_ms` | `float` | Median search latency |
| `p95_latency_ms` | `float` | 95th-percentile search latency |
| `p99_latency_ms` | `float` | 99th-percentile search latency |
| `by_strategy` | `dict[SearchStrategy, StrategyStats]` | Per-tier breakdown |

---

## Hit Rate

$$\text{hit\_rate} = \frac{\text{total\_hits}}{\text{total\_hits} + \text{total\_misses}}$$

A hit rate of 0.8 means 80% of LLM calls were avoided.

---

## Latency Percentiles

Search latency is tracked per request. Percentiles are computed over a rolling window:

```python
stats = await cache.get_stats()
print(f"P50: {stats.p50_latency_ms:.1f} ms")
print(f"P95: {stats.p95_latency_ms:.1f} ms")
print(f"P99: {stats.p99_latency_ms:.1f} ms")
```

Expected ranges (in-memory backend, FastEmbed):

| Tier | P50 | P95 |
|---|---|---|
| L1 Cache | < 0.1 ms | < 0.5 ms |
| Template Match | 1–3 ms | 5 ms |
| Exact / Semantic Vector | 5–15 ms | 20 ms |
| Fuzzy | 20–40 ms | 50 ms |

---

## Per-Strategy Breakdown

`stats.by_strategy` maps each `SearchStrategy` to a `StrategyStats` object:

```python
from medha.types import SearchStrategy

stats = await cache.get_stats()
for strategy, s in stats.by_strategy.items():
    print(f"{strategy.name}: hits={s.hits}, avg={s.avg_latency_ms:.1f} ms")
```

Output example:

```
L1_CACHE: hits=142, avg=0.08 ms
TEMPLATE_MATCH: hits=38, avg=2.1 ms
EXACT_VECTOR_MATCH: hits=21, avg=8.3 ms
SEMANTIC_MATCH: hits=64, avg=11.2 ms
FUZZY_MATCH: hits=5, avg=31.4 ms
```

---

## Logging

Use `setup_logging()` to configure the `medha` logger:

```python
from medha.logging import setup_logging

# Human-readable text format
setup_logging(level="INFO", format="text")

# Structured JSON for log aggregation (Datadog, CloudWatch, etc.)
setup_logging(level="INFO", format="json")
```

Or configure the `medha` logger directly:

```python
import logging

logging.getLogger("medha").setLevel(logging.DEBUG)
```

Key log events:

| Event | Level | Description |
|---|---|---|
| `cache.hit` | INFO | A search returned a cache hit |
| `cache.miss` | INFO | A search returned no result |
| `backend.init` | INFO | Backend connected successfully |
| `backend.error` | ERROR | Backend connection or query failed |
| `cleanup.run` | DEBUG | Background cleanup sweep started |
| `cleanup.deleted` | DEBUG | Number of expired entries removed |

---

## Prometheus Integration

Medha does not ship a Prometheus exporter, but `CacheStats` is easy to bridge:

```python
import asyncio
from prometheus_client import Counter, Histogram, start_http_server
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

hits_counter = Counter("medha_hits_total", "Cache hits", ["strategy"])
misses_counter = Counter("medha_misses_total", "Cache misses")
latency_hist = Histogram(
    "medha_search_latency_seconds",
    "Search latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


async def search_with_metrics(cache, question: str):
    import time

    t0 = time.perf_counter()
    hit = await cache.search(question)
    elapsed = time.perf_counter() - t0
    latency_hist.observe(elapsed)

    if hit:
        hits_counter.labels(strategy=hit.strategy.name).inc()
    else:
        misses_counter.inc()

    return hit


async def main():
    start_http_server(8000)  # expose /metrics on :8000
    settings = Settings(backend_type="memory")
    async with Medha("demo", embedder=FastEmbedAdapter(), settings=settings) as cache:
        while True:
            await search_with_metrics(cache, "How many users?")
            await asyncio.sleep(1)


asyncio.run(main())
```

Access metrics at `http://localhost:8000/metrics`.
