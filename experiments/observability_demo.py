"""Demo: Medha observability API — CacheStats snapshots during a realistic workload.

Not a performance benchmark. Demonstrates how to read CacheStats at runtime:
hit_rate, miss_rate, avg latency, p95, and per-strategy breakdown.

Usage:
    python experiments/observability_demo.py
    python experiments/observability_demo.py --queries 500 --cache-size 200 --print-every 100
    python experiments/observability_demo.py --queries 1000 --print-every 200 --output obs_report.json
"""

import asyncio
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset
from cost_benchmark import build_workload


def _fmt_snapshot(idx: int, total: int, stats) -> str:
    hit_pct = stats.hit_rate * 100
    miss_pct = stats.miss_rate * 100
    avg_ms = stats.avg_latency_ms
    p95_ms = stats.p95_latency_ms
    return (
        f"[Query {idx}/{total}]  "
        f"hit_rate={hit_pct:.1f}%  miss={miss_pct:.1f}%  "
        f"avg={avg_ms:.1f}ms  p95={p95_ms:.1f}ms"
    )


def _fmt_strategy_breakdown(stats) -> str:
    lines = []
    for strategy, s in sorted(stats.by_strategy.items()):
        pct = (s.count / stats.total_requests * 100) if stats.total_requests > 0 else 0.0
        lines.append(f"    {strategy:<22}  {s.count:>6}  ({pct:5.1f}%)")
    return "\n".join(lines)


async def run_demo(
    num_queries: int,
    cache_size: int,
    print_every: int,
) -> list[dict]:
    embedder = FastEmbedAdapter()
    settings = Settings(backend_type="memory", collect_stats=True)
    medha = Medha("obs_demo", embedder=embedder, settings=settings)
    await medha.start()

    # Warm-up
    print(f"Warming up: storing {cache_size} entries...")
    dataset = generate_dataset(cache_size, lang="sql")
    entries = [{"question": d["question"], "generated_query": d["query"]} for d in dataset]
    await medha.store_many(entries)

    # Build workload (40% exact, 30% paraphrase, 30% novel)
    workload = build_workload(dataset, num_queries)
    print(f"Running {num_queries} queries (40% exact / 30% paraphrase / 30% novel)...\n")

    snapshots = []

    for i, question in enumerate(workload, start=1):
        await medha.search(question)

        if i % print_every == 0 or i == num_queries:
            stats = await medha.stats()
            line = _fmt_snapshot(i, num_queries, stats)
            print(line)
            breakdown = _fmt_strategy_breakdown(stats)
            if breakdown:
                print(breakdown)
            print()

            snapshots.append({
                "query_index": i,
                "total_requests": stats.total_requests,
                "total_hits": stats.total_hits,
                "total_misses": stats.total_misses,
                "hit_rate_pct": round(stats.hit_rate * 100, 2),
                "miss_rate_pct": round(stats.miss_rate * 100, 2),
                "avg_latency_ms": round(stats.avg_latency_ms, 2),
                "p50_latency_ms": round(stats.p50_latency_ms, 2),
                "p95_latency_ms": round(stats.p95_latency_ms, 2),
                "p99_latency_ms": round(stats.p99_latency_ms, 2),
                "by_strategy": {
                    k: {"count": v.count, "avg_latency_ms": round(v.avg_latency_ms, 2)}
                    for k, v in stats.by_strategy.items()
                },
            })

    await medha.close()
    return snapshots


def main():
    parser = argparse.ArgumentParser(
        description="Demo Medha observability API with periodic CacheStats snapshots.",
    )
    parser.add_argument(
        "--queries", type=int, default=500,
        help="Total number of queries to run (default: 500)",
    )
    parser.add_argument(
        "--cache-size", type=int, default=200,
        help="Number of entries to pre-populate in the cache (default: 200)",
    )
    parser.add_argument(
        "--print-every", type=int, default=100,
        help="Print a stats snapshot every N queries (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save all snapshots to this JSON file (optional)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("Medha Observability Demo")
    print(f"  Queries: {args.queries}  Cache size: {args.cache_size}  "
          f"Print every: {args.print_every}\n")

    snapshots = asyncio.run(run_demo(args.queries, args.cache_size, args.print_every))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(snapshots, f, indent=2)
        print(f"Snapshots saved to {args.output}")


if __name__ == "__main__":
    main()
