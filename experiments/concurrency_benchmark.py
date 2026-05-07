"""Benchmark: embedding deduplication under concurrent load.

Verifies and measures the _pending_embeddings deduplication mechanism in
Medha.  When N coroutines concurrently search the same question, only one
embedding call should be made to the embedder; the rest join the in-flight
Future and share the result.

Metrics collected:
    - Actual embedder calls vs expected (without deduplication: N calls)
    - Wall-clock latency for the entire concurrent batch
    - Per-query latency distribution
    - Throughput (queries/second)

Usage:
    python experiments/concurrency_benchmark.py --concurrency 50
    python experiments/concurrency_benchmark.py --concurrency 100 --unique-questions 10 --output conc_report.json
"""

import asyncio
import argparse
import json
import time
from statistics import mean, median, stdev
from unittest.mock import AsyncMock, patch

from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset


# ---------------------------------------------------------------------------
# Instrumented embedder wrapper
# ---------------------------------------------------------------------------

class CountingEmbedder:
    """Wraps FastEmbedAdapter and counts actual aembed() calls made."""

    def __init__(self):
        self._inner = FastEmbedAdapter()
        self.call_count = 0
        self._lock = asyncio.Lock()

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def aembed(self, text: str) -> list[float]:
        async with self._lock:
            self.call_count += 1
        return await self._inner.aembed(text)

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._lock:
            self.call_count += len(texts)
        return await self._inner.aembed_batch(texts)

    def reset(self):
        self.call_count = 0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_concurrent_batch(
    medha: Medha,
    questions: list[str],
) -> tuple[list[float], float]:
    """Launch all searches concurrently and collect latencies.

    Returns:
        (per-query latency list in ms, total wall-clock time in ms)
    """
    latencies: list[float] = [0.0] * len(questions)

    async def _search(idx: int, q: str) -> None:
        t = time.perf_counter()
        await medha.search(q)
        latencies[idx] = (time.perf_counter() - t) * 1000

    wall_start = time.perf_counter()
    await asyncio.gather(*[_search(i, q) for i, q in enumerate(questions)])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    return latencies, wall_ms


async def run_benchmark(
    concurrency: int,
    unique_questions: int,
    seed: int | None = None,
) -> dict:
    import random
    if seed is not None:
        random.seed(seed)

    # Generate a small unique set, then repeat to fill concurrency slots
    dataset = generate_dataset(unique_questions, lang="sql")
    questions_pool = [e["question"] for e in dataset]

    # Build the concurrent workload: repeat questions so duplicates are common
    concurrent_questions = [
        questions_pool[i % unique_questions] for i in range(concurrency)
    ]

    embedder = CountingEmbedder()
    # backend_type="memory" — pure-Python backend (default in 0.3.1).
    # This benchmark measures embedding deduplication, not backend performance.
    settings = Settings(backend_type="memory")
    medha = Medha("conc_bench", embedder=embedder, settings=settings)
    await medha.start()

    # Populate the cache (uses store, which embeds once per unique question)
    print(f"  Populating cache with {unique_questions} unique entries...")
    embedder.reset()
    for entry in dataset:
        await medha.store(entry["question"], entry["query"])
    store_embed_calls = embedder.call_count

    # Clear in-process caches to ensure embedder is called during search
    await medha.clear_caches()
    embedder.reset()

    print(f"  Launching {concurrency} concurrent searches "
          f"({unique_questions} unique questions)...")
    latencies, wall_ms = await run_concurrent_batch(medha, concurrent_questions)

    search_embed_calls = embedder.call_count

    await medha.close()

    # Without deduplication each unique question would be embedded concurrency/unique times
    duplicates_in_workload = concurrency - unique_questions
    dedup_ratio = (
        round(duplicates_in_workload / search_embed_calls, 2)
        if search_embed_calls > 0 else 0.0
    )

    stats = {}
    if latencies:
        s = sorted(latencies)
        p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
        stats = {
            "mean_ms":   round(mean(latencies), 3),
            "median_ms": round(median(latencies), 3),
            "p95_ms":    round(p95, 3),
            "stdev_ms":  round(stdev(latencies), 3) if len(latencies) > 1 else 0.0,
        }

    throughput = round(concurrency / (wall_ms / 1000), 1) if wall_ms > 0 else 0.0

    return {
        "concurrency": concurrency,
        "unique_questions": unique_questions,
        "concurrent_workload_size": len(concurrent_questions),
        "store_embed_calls": store_embed_calls,
        "search_embed_calls_actual": search_embed_calls,
        "search_embed_calls_without_dedup": concurrency,
        "calls_saved_by_dedup": max(0, concurrency - search_embed_calls),
        "dedup_ratio": dedup_ratio,
        "wall_time_ms": round(wall_ms, 2),
        "throughput_qps": throughput,
        "latency": stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    print(f"\n{'='*60}")
    print("Concurrency / Deduplication Benchmark Results")
    print(f"{'='*60}")
    print(f"  Concurrency:         {results['concurrency']} queries at once")
    print(f"  Unique questions:    {results['unique_questions']}")
    print(f"  Embed calls (store): {results['store_embed_calls']}")
    print(f"  Embed calls (search, actual):   {results['search_embed_calls_actual']}")
    print(f"  Embed calls (search, no dedup): {results['search_embed_calls_without_dedup']}")
    saved = results['calls_saved_by_dedup']
    total_without = results['search_embed_calls_without_dedup']
    pct = round(saved / total_without * 100, 1) if total_without > 0 else 0.0
    print(f"  Calls saved by dedup: {saved} ({pct}%)")
    print(f"  Dedup ratio:          {results['dedup_ratio']:.2f}x")
    print()
    print(f"  Wall-clock time:    {results['wall_time_ms']:.2f}ms")
    print(f"  Throughput:         {results['throughput_qps']:.1f} queries/sec")
    s = results['latency']
    print(f"  Per-query latency:  mean={s['mean_ms']:.2f}ms  "
          f"median={s['median_ms']:.2f}ms  p95={s['p95_ms']:.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Medha embedding deduplication under concurrent load",
    )
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Number of concurrent search coroutines (default: 50)")
    parser.add_argument("--unique-questions", type=int, default=10,
                        help="Number of unique questions in the workload (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (prints to stdout if omitted)")
    args = parser.parse_args()

    print("Medha Concurrency / Deduplication Benchmark")
    print(f"  Concurrency: {args.concurrency}, Unique questions: {args.unique_questions}")

    results = asyncio.run(run_benchmark(args.concurrency, args.unique_questions, args.seed))
    _print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
