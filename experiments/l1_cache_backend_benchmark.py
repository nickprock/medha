"""Benchmark: InMemoryL1Cache vs RedisL1Cache.

Compares latency, hit rate, and throughput between the two L1 cache backends
across workloads with varying repeat ratios.

The Redis scenario requires a running Redis instance.  If Redis is not
available the script skips that scenario and reports only in-memory results.

Usage:
    python experiments/l1_cache_backend_benchmark.py
    python experiments/l1_cache_backend_benchmark.py --redis-url redis://localhost:6379/1
    python experiments/l1_cache_backend_benchmark.py --output l1_report.json
"""

import asyncio
import argparse
import json
import time
from statistics import mean, median, stdev

from medha import Medha, Settings, InMemoryL1Cache
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(latencies: list[float]) -> dict:
    if not latencies:
        return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0, "stdev_ms": 0}
    s = sorted(latencies)
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    return {
        "mean_ms":   round(mean(latencies), 3),
        "median_ms": round(median(latencies), 3),
        "p95_ms":    round(p95, 3),
        "stdev_ms":  round(stdev(latencies), 3) if len(latencies) > 1 else 0.0,
    }


def _build_workload(dataset: list[dict], repeat_ratio: float, total: int) -> list[str]:
    """Build a workload with a given fraction of repeated questions."""
    import random
    n_repeat = int(total * repeat_ratio)
    n_novel = total - n_repeat

    repeated = [random.choice(dataset)["question"] for _ in range(n_repeat)]
    # Novel questions are paraphrased to avoid L1 hits (different hash)
    novel = [
        f"Tell me about {random.choice(dataset)['question'].lower()}"
        for _ in range(n_novel)
    ]
    workload = repeated + novel
    random.shuffle(workload)
    return workload


async def _run_workload(medha: Medha, workload: list[str]) -> tuple[list[float], int]:
    """Run workload and return (latencies_ms, hit_count)."""
    from medha.types import SearchStrategy
    latencies = []
    hits = 0
    for q in workload:
        t = time.perf_counter()
        result = await medha.search(q)
        latencies.append((time.perf_counter() - t) * 1000)
        if result.strategy not in (SearchStrategy.NO_MATCH, SearchStrategy.ERROR):
            hits += 1
    return latencies, hits


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

async def run_in_memory_scenario(
    dataset: list[dict],
    workloads: dict[str, list[str]],
    max_size: int,
) -> dict:
    """Run all workloads using InMemoryL1Cache."""
    results = {}
    for label, workload in workloads.items():
        embedder = FastEmbedAdapter()
        # backend_type="memory" — pure-Python vector backend; this benchmark
        # measures L1 cache behaviour, not the vector backend performance.
        settings = Settings(backend_type="memory", l1_cache_max_size=max_size)
        l1 = InMemoryL1Cache(max_size=max_size)
        medha = Medha("in_memory_bench", embedder=embedder, settings=settings, l1_backend=l1)
        await medha.start()

        for entry in dataset:
            await medha.store(entry["question"], entry["query"])

        # Clear caches so we measure fresh behaviour (not post-store L1 warmth)
        await medha.clear_caches()

        latencies, hits = await _run_workload(medha, workload)
        await medha.close()

        results[label] = {
            **_stats(latencies),
            "hit_rate_pct": round(hits / len(workload) * 100, 1) if workload else 0.0,
        }
    return results


async def run_redis_scenario(
    dataset: list[dict],
    workloads: dict[str, list[str]],
    redis_url: str,
    ttl: int | None,
) -> dict | None:
    """Run all workloads using RedisL1Cache.  Returns None if Redis unavailable."""
    try:
        from medha import RedisL1Cache
        import redis.asyncio as aioredis  # type: ignore[import]
        client = aioredis.from_url(redis_url)
        await client.ping()
        await client.aclose()
    except Exception as exc:
        print(f"  Redis not available ({exc}) — skipping Redis scenario")
        return None

    results = {}
    for label, workload in workloads.items():
        embedder = FastEmbedAdapter()
        settings = Settings(backend_type="memory")
        l1 = RedisL1Cache(url=redis_url, prefix=f"medha:bench:{label}", ttl=ttl)
        medha = Medha("redis_bench", embedder=embedder, settings=settings, l1_backend=l1)
        await medha.start()

        for entry in dataset:
            await medha.store(entry["question"], entry["query"])

        await medha.clear_caches()

        latencies, hits = await _run_workload(medha, workload)
        await medha.close()

        results[label] = {
            **_stats(latencies),
            "hit_rate_pct": round(hits / len(workload) * 100, 1) if workload else 0.0,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(
    cache_size: int,
    num_queries: int,
    redis_url: str,
    redis_ttl: int | None,
    seed: int | None = None,
) -> dict:
    import random
    if seed is not None:
        random.seed(seed)

    print(f"  Generating {cache_size} dataset entries...")
    dataset = generate_dataset(cache_size, lang="sql")

    # Three workloads: low / medium / high repeat ratio
    workloads = {
        "low_repeat_20pct":    _build_workload(dataset, 0.20, num_queries),
        "medium_repeat_60pct": _build_workload(dataset, 0.60, num_queries),
        "high_repeat_90pct":   _build_workload(dataset, 0.90, num_queries),
    }

    print("  Running InMemoryL1Cache scenarios...")
    in_memory = await run_in_memory_scenario(dataset, workloads, cache_size)

    print("  Running RedisL1Cache scenarios...")
    redis = await run_redis_scenario(dataset, workloads, redis_url, redis_ttl)

    return {
        "cache_size": cache_size,
        "num_queries": num_queries,
        "workload_labels": list(workloads.keys()),
        "in_memory": in_memory,
        "redis": redis,
    }


def _print_summary(results: dict) -> None:
    print(f"\n{'='*60}")
    print("L1 Cache Backend Benchmark Results")
    print(f"{'='*60}")
    print(f"  Cache size: {results['cache_size']}, Queries: {results['num_queries']}")
    print()

    for backend_label, data in [("InMemoryL1Cache", results["in_memory"]),
                                  ("RedisL1Cache",    results["redis"])]:
        if data is None:
            print(f"  {backend_label}: skipped (not available)")
            continue
        print(f"  {backend_label}:")
        for workload, s in data.items():
            print(f"    {workload:30s}  hit={s['hit_rate_pct']:5.1f}%  "
                  f"mean={s['mean_ms']:7.3f}ms  p95={s['p95_ms']:7.3f}ms")

    # Comparison: Redis overhead vs in-memory
    if results["redis"]:
        print("\n  Redis vs In-Memory overhead (mean_ms delta):")
        for wl in results["workload_labels"]:
            im = results["in_memory"][wl]["mean_ms"]
            rd = results["redis"][wl]["mean_ms"]
            delta = round(rd - im, 3)
            print(f"    {wl:30s}  +{delta:.3f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Medha L1 cache: InMemory vs Redis",
    )
    parser.add_argument("--cache-size", type=int, default=300,
                        help="Dataset / cache size (default: 300)")
    parser.add_argument("--num-queries", type=int, default=500,
                        help="Queries per workload (default: 500)")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0",
                        help="Redis connection URL (default: redis://localhost:6379/0)")
    parser.add_argument("--redis-ttl", type=int, default=None,
                        help="Redis key TTL in seconds (default: none)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (prints to stdout if omitted)")
    args = parser.parse_args()

    print("Medha L1 Cache Backend Benchmark")
    print(f"  Cache size: {args.cache_size}, Queries: {args.num_queries}")
    print(f"  Redis URL: {args.redis_url}")

    results = asyncio.run(run_benchmark(
        args.cache_size, args.num_queries,
        args.redis_url, args.redis_ttl, args.seed,
    ))
    _print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
