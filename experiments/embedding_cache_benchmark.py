"""Benchmark: persistent embedding cache warm-start vs cold-start.

Measures the latency gain when embeddings are pre-loaded from disk
(warm-start) compared to computing them from scratch (cold-start).
Also measures the benefit of the in-process LRU embedding cache on
repeated searches within the same session.

Usage:
    python experiments/embedding_cache_benchmark.py --size 500
    python experiments/embedding_cache_benchmark.py --size 1000 --output emb_cache_report.json
"""

import asyncio
import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev

from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _stats(latencies: list[float]) -> dict:
    if not latencies:
        return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0, "stdev_ms": 0}
    s = sorted(latencies)
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    return {
        "mean_ms": round(mean(latencies), 3),
        "median_ms": round(median(latencies), 3),
        "p95_ms": round(p95, 3),
        "stdev_ms": round(stdev(latencies), 3) if len(latencies) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

async def _run_searches(medha: Medha, questions: list[str]) -> list[float]:
    """Run searches and return per-query latencies in ms."""
    latencies = []
    for q in questions:
        t = time.perf_counter()
        await medha.search(q)
        latencies.append((time.perf_counter() - t) * 1000)
    return latencies


async def scenario_cold_start(
    dataset: list[dict],
    questions: list[str],
) -> dict:
    """Cold-start: no persistent cache, embeddings computed from scratch every time."""
    embedder = FastEmbedAdapter()
    # backend_type="memory" — pure-Python backend; this benchmark measures the
    # embedding cache, not backend performance.
    settings = Settings(backend_type="memory")
    medha = Medha("cold_start", embedder=embedder, settings=settings)
    await medha.start()

    for entry in dataset:
        await medha.store(entry["question"], entry["query"])

    # Clear in-process LRU so embedding cache is cold
    await medha.clear_caches()

    latencies = await _run_searches(medha, questions)
    await medha.close()
    return _stats(latencies)


async def scenario_warm_start(
    dataset: list[dict],
    questions: list[str],
    cache_path: str,
) -> dict:
    """Warm-start: embeddings loaded from disk before searching."""
    embedder = FastEmbedAdapter()
    settings = Settings(backend_type="memory", embedding_cache_path=cache_path)
    medha = Medha("warm_start", embedder=embedder, settings=settings)
    await medha.start()

    for entry in dataset:
        await medha.store(entry["question"], entry["query"])

    # Save embedding cache to disk then close
    await medha.close()

    # Re-open: embeddings loaded from disk → warm-start
    medha2 = Medha("warm_start", embedder=embedder, settings=settings)
    await medha2.start()

    # Re-populate vector store (simulates a restart where vector data was re-loaded)
    for entry in dataset:
        await medha2.store(entry["question"], entry["query"])

    latencies = await _run_searches(medha2, questions)
    await medha2.close()
    return _stats(latencies)


async def scenario_lru_hit(
    dataset: list[dict],
    questions: list[str],
) -> dict:
    """In-process LRU: search the same questions twice, measure second-pass latency."""
    embedder = FastEmbedAdapter()
    settings = Settings(backend_type="memory")
    medha = Medha("lru_hit", embedder=embedder, settings=settings)
    await medha.start()

    for entry in dataset:
        await medha.store(entry["question"], entry["query"])

    # First pass: populate LRU embedding cache
    for q in questions:
        await medha.search(q)

    # Second pass: embeddings served from LRU
    latencies = await _run_searches(medha, questions)
    await medha.close()
    return _stats(latencies)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(size: int, seed: int | None = None) -> dict:
    import random
    if seed is not None:
        random.seed(seed)

    print(f"Generating {size} dataset entries...")
    dataset = generate_dataset(size, lang="sql")

    # Use the dataset questions as the query workload (realistic repeat pattern)
    questions = [e["question"] for e in dataset]

    results = {}

    print("Running scenario: cold-start...")
    results["cold_start"] = await scenario_cold_start(dataset, questions)
    print(f"  mean={results['cold_start']['mean_ms']:.2f}ms")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name

    try:
        print("Running scenario: warm-start (disk cache)...")
        results["warm_start"] = await scenario_warm_start(dataset, questions, cache_path)
        print(f"  mean={results['warm_start']['mean_ms']:.2f}ms")
    finally:
        if os.path.exists(cache_path):
            os.unlink(cache_path)

    print("Running scenario: in-process LRU hit...")
    results["lru_hit"] = await scenario_lru_hit(dataset, questions)
    print(f"  mean={results['lru_hit']['mean_ms']:.2f}ms")

    # Compute speedup ratios relative to cold-start
    cold_mean = results["cold_start"]["mean_ms"] or 1.0
    results["speedup"] = {
        "warm_vs_cold": round(cold_mean / (results["warm_start"]["mean_ms"] or cold_mean), 2),
        "lru_vs_cold":  round(cold_mean / (results["lru_hit"]["mean_ms"] or cold_mean), 2),
    }
    results["dataset_size"] = size
    results["num_queries"] = len(questions)
    return results


def _print_summary(results: dict) -> None:
    print(f"\n{'='*60}")
    print("Embedding Cache Benchmark Results")
    print(f"{'='*60}")
    print(f"  Dataset size:  {results['dataset_size']} entries")
    print(f"  Queries run:   {results['num_queries']}")
    print()
    for scenario in ("cold_start", "warm_start", "lru_hit"):
        s = results[scenario]
        label = scenario.replace("_", " ").title()
        print(f"  {label:30s}  mean={s['mean_ms']:7.3f}ms  "
              f"median={s['median_ms']:7.3f}ms  p95={s['p95_ms']:7.3f}ms")

    print(f"\n  Speedup (warm-start vs cold):    {results['speedup']['warm_vs_cold']:.2f}x")
    print(f"  Speedup (in-process LRU vs cold): {results['speedup']['lru_vs_cold']:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Medha embedding cache: cold vs warm vs LRU",
    )
    parser.add_argument("--size", type=int, default=500,
                        help="Dataset / cache size (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (prints to stdout if omitted)")
    args = parser.parse_args()

    print("Medha Embedding Cache Benchmark")
    print(f"  Cache size: {args.size}")

    results = asyncio.run(run_benchmark(args.size, args.seed))
    _print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
