"""Benchmark: ingestion throughput — sequential vs store_batch vs store_many.

Compares entries/second across three ingestion strategies:
    sequential     N individual store() calls — baseline
    store_batch    Single aembed_batch() + bulk upsert via medha.store_batch()
    store_many(C)  Chunked aembed_batch with batch_embed_concurrency=C via medha.store_many()

Usage:
    python experiments/batch_benchmark.py
    python experiments/batch_benchmark.py --size 1000 --concurrencies 1,2,4,8
    python experiments/batch_benchmark.py --size 500 --seed 42 --output batch_report.json

Output:
    Medha Batch Ingestion Benchmark
      Dataset size: 500
      Strategy                     Total(s)  Throughput(eps)
      sequential                   12.340    40.5
      store_batch                   3.210    155.8
      store_many(concurrency=1)     3.180    157.2
      store_many(concurrency=2)     1.890    264.6
      store_many(concurrency=4)     1.120    446.4
      Speedup store_batch vs seq:   3.84x
      Speedup store_many(c=4) vs seq: 11.01x
"""

import asyncio
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset


def _make_batch(dataset: list[dict]) -> list[dict]:
    return [
        {"question": d["question"], "generated_query": d["query"]}
        for d in dataset
    ]


async def scenario_sequential(dataset: list[dict]) -> dict:
    """Baseline: one store() call per entry."""
    embedder = FastEmbedAdapter()
    settings = Settings(backend_type="memory")
    medha = Medha("batch_seq", embedder=embedder, settings=settings)
    await medha.start()

    entries = _make_batch(dataset)
    t0 = time.perf_counter()
    for entry in entries:
        await medha.store(entry["question"], entry["generated_query"])
    total_s = time.perf_counter() - t0

    await medha.close()
    return {
        "strategy": "sequential",
        "entries": len(entries),
        "total_s": round(total_s, 3),
        "throughput_eps": round(len(entries) / total_s, 1),
    }


async def scenario_store_batch(dataset: list[dict]) -> dict:
    """Single aembed_batch() + bulk upsert via store_batch()."""
    embedder = FastEmbedAdapter()
    settings = Settings(backend_type="memory")
    medha = Medha("batch_sb", embedder=embedder, settings=settings)
    await medha.start()

    entries = _make_batch(dataset)
    t0 = time.perf_counter()
    await medha.store_batch(entries)
    total_s = time.perf_counter() - t0

    await medha.close()
    return {
        "strategy": "store_batch",
        "entries": len(entries),
        "total_s": round(total_s, 3),
        "throughput_eps": round(len(entries) / total_s, 1),
    }


async def scenario_store_many(dataset: list[dict], concurrency: int) -> dict:
    """Chunked aembed_batch with configurable embedding concurrency."""
    embedder = FastEmbedAdapter()
    settings = Settings(backend_type="memory", batch_embed_concurrency=concurrency)
    medha = Medha("batch_sm", embedder=embedder, settings=settings)
    await medha.start()

    entries = _make_batch(dataset)
    t0 = time.perf_counter()
    await medha.store_many(entries)
    total_s = time.perf_counter() - t0

    await medha.close()
    return {
        "strategy": f"store_many(concurrency={concurrency})",
        "entries": len(entries),
        "total_s": round(total_s, 3),
        "throughput_eps": round(len(entries) / total_s, 1),
    }


async def run_all(size: int, concurrencies: list[int]) -> list[dict]:
    dataset = generate_dataset(size, lang="sql")

    print(f"\nMedha Batch Ingestion Benchmark")
    print(f"  Dataset size: {size}")
    print(f"  {'Strategy':<32} {'Total(s)':>10}  {'Throughput(eps)':>15}")
    print(f"  {'-'*62}")

    results = []

    seq = await scenario_sequential(dataset)
    results.append(seq)
    print(f"  {seq['strategy']:<32} {seq['total_s']:>10.3f}  {seq['throughput_eps']:>15.1f}")

    sb = await scenario_store_batch(dataset)
    results.append(sb)
    print(f"  {sb['strategy']:<32} {sb['total_s']:>10.3f}  {sb['throughput_eps']:>15.1f}")

    last_sm = None
    for c in concurrencies:
        sm = await scenario_store_many(dataset, c)
        results.append(sm)
        last_sm = sm
        print(f"  {sm['strategy']:<32} {sm['total_s']:>10.3f}  {sm['throughput_eps']:>15.1f}")

    seq_eps = seq["throughput_eps"]
    if seq_eps > 0:
        print(f"\n  Speedup store_batch vs seq:          {sb['throughput_eps'] / seq_eps:.2f}x")
        if last_sm:
            print(f"  Speedup {last_sm['strategy']} vs seq: {last_sm['throughput_eps'] / seq_eps:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Medha batch ingestion throughput.",
    )
    parser.add_argument(
        "--size", type=int, default=500,
        help="Number of entries to ingest (default: 500)",
    )
    parser.add_argument(
        "--concurrencies", type=str, default="1,2,4",
        help="Comma-separated concurrency values for store_many (default: 1,2,4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",")]

    results = asyncio.run(run_all(args.size, concurrencies))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
