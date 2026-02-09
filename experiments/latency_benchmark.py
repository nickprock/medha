"""Benchmark Medha search latency.

Measures:
    - L1 cache hit time (Tier 0)
    - Exact vector match time (Tier 2)
    - Semantic similarity time (Tier 3)
    - Miss time (no match)

At configurable collection sizes (default: 100, 1000, 10000).

Usage:
    python experiments/latency_benchmark.py --sizes 100,1000,10000
    python experiments/latency_benchmark.py --sizes 100,500 --num-queries 50 --output latency_report.json
"""

import asyncio
import argparse
import json
import random
import time
from statistics import mean, median, stdev

from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset


# ---------------------------------------------------------------------------
# Paraphrase & unrelated question generators
# ---------------------------------------------------------------------------

_PARAPHRASE_PREFIXES = [
    "Can you tell me ",
    "I'd like to know ",
    "Please show me ",
    "I want to see ",
    "Could you find ",
]

_PARAPHRASE_SUFFIXES = [
    "",
    " please",
    " for me",
    " if possible",
]

_SYNONYMS = {
    "How many": "What is the count of",
    "Show the top": "Display the first",
    "List all": "Show every",
    "What is the average": "What's the mean",
    "Find": "Retrieve",
}


def paraphrase(entries: list[dict]) -> list[str]:
    """Generate paraphrased versions of dataset questions.

    Applies lightweight transformations: synonym substitution, prefix/suffix
    additions. These won't be exact matches but should be semantically close
    enough to trigger Tier 3 (semantic match).
    """
    paraphrased = []
    for entry in entries:
        question = entry["question"]

        # Try synonym substitution first
        modified = question
        for original, replacement in _SYNONYMS.items():
            if question.startswith(original):
                modified = question.replace(original, replacement, 1)
                break
        else:
            # No synonym matched — use prefix/suffix wrapping
            prefix = random.choice(_PARAPHRASE_PREFIXES)
            suffix = random.choice(_PARAPHRASE_SUFFIXES)
            # Lowercase the first char when prepending a prefix
            lowered = question[0].lower() + question[1:] if question else question
            modified = f"{prefix}{lowered}{suffix}"

        paraphrased.append(modified)

    return paraphrased


_UNRELATED_QUESTIONS = [
    "What is the speed of light?",
    "Who painted the Mona Lisa?",
    "How do I make pasta carbonara?",
    "What year did the Berlin Wall fall?",
    "Explain quantum entanglement",
    "What is the capital of Mongolia?",
    "How tall is Mount Everest?",
    "Who wrote Crime and Punishment?",
    "What causes the northern lights?",
    "How many moons does Jupiter have?",
    "What is the boiling point of mercury?",
    "Who invented the telephone?",
    "What is photosynthesis?",
    "How does a combustion engine work?",
    "What are the symptoms of altitude sickness?",
    "Who was the first person in space?",
    "What is the Fibonacci sequence?",
    "How do birds navigate during migration?",
    "What is the deepest point in the ocean?",
    "Who discovered penicillin?",
]


def generate_unrelated(count: int) -> list[str]:
    """Generate questions unrelated to any stored dataset entry.

    These should produce cache misses (no match) at every tier.
    """
    questions = []
    for i in range(count):
        base = _UNRELATED_QUESTIONS[i % len(_UNRELATED_QUESTIONS)]
        if i >= len(_UNRELATED_QUESTIONS):
            # Add a numeric suffix to ensure uniqueness beyond the pool size
            base = f"{base} (variant {i})"
        questions.append(base)
    return questions


# ---------------------------------------------------------------------------
# Latency measurement helpers
# ---------------------------------------------------------------------------

def _compute_stats(latencies: list[float]) -> dict:
    """Compute mean, median, p95, and stdev from a list of latencies in ms."""
    if not latencies:
        return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0, "stdev_ms": 0}

    sorted_lats = sorted(latencies)
    p95_idx = min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)

    return {
        "mean_ms": round(mean(latencies), 2),
        "median_ms": round(median(latencies), 2),
        "p95_ms": round(sorted_lats[p95_idx], 2),
        "stdev_ms": round(stdev(latencies), 2) if len(latencies) > 1 else 0,
    }


async def measure_tier(medha: Medha, queries: list) -> dict:
    """Measure latency for a set of queries."""
    latencies = []
    for q in queries:
        question = q if isinstance(q, str) else q["question"]
        start = time.perf_counter()
        await medha.search(question)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    return _compute_stats(latencies)


async def measure_l1(medha: Medha, entries: list[dict]) -> dict:
    """Measure L1 cache latency.

    For each entry: search once (populates L1), then search again (L1 hit).
    Only the second call's latency is recorded.
    """
    latencies = []
    for entry in entries:
        question = entry["question"]
        # First call — populates L1 cache
        await medha.search(question)
        # Second call — should hit L1 cache
        start = time.perf_counter()
        await medha.search(question)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    return _compute_stats(latencies)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def benchmark(
    collection_sizes: list[int],
    num_queries: int = 100,
    seed: int | None = None,
) -> dict:
    """Run the latency benchmark across collection sizes."""
    if seed is not None:
        random.seed(seed)

    results = {}

    for size in collection_sizes:
        print(f"\n{'='*50}")
        print(f"Collection size: {size}")
        print(f"{'='*50}")

        # --- Setup ---
        embedder = FastEmbedAdapter()
        settings = Settings(qdrant_mode="memory")
        medha = Medha(f"bench_{size}", embedder=embedder, settings=settings)
        await medha.start()

        # --- Populate ---
        print(f"  Generating {size} pairs...")
        dataset = generate_dataset(size, lang="sql")

        print(f"  Storing {size} entries...")
        batch = [
            {"question": d["question"], "generated_query": d["query"]}
            for d in dataset
        ]
        await medha.store_batch(batch)

        # Limit queries to what's available
        query_count = min(num_queries, len(dataset))
        sample = dataset[:query_count]

        # --- Tier 2: Exact match (search with stored questions) ---
        # Clear L1 first so we measure vector search, not L1 hits
        medha.clear_caches()
        print(f"  Measuring exact match ({query_count} queries)...")
        tier_results = {}
        tier_results["exact_match"] = await measure_tier(medha, sample)
        print(f"    mean={tier_results['exact_match']['mean_ms']:.2f}ms")

        # --- Tier 3: Semantic match (search with paraphrased questions) ---
        medha.clear_caches()
        print(f"  Measuring semantic match ({query_count} queries)...")
        paraphrased = paraphrase(sample)
        tier_results["semantic_match"] = await measure_tier(medha, paraphrased)
        print(f"    mean={tier_results['semantic_match']['mean_ms']:.2f}ms")

        # --- No match (search with unrelated questions) ---
        medha.clear_caches()
        print(f"  Measuring no-match ({query_count} queries)...")
        unrelated = generate_unrelated(query_count)
        tier_results["no_match"] = await measure_tier(medha, unrelated)
        print(f"    mean={tier_results['no_match']['mean_ms']:.2f}ms")

        # --- Tier 0: L1 cache (must be last — it populates L1) ---
        medha.clear_caches()
        print(f"  Measuring L1 cache ({query_count} queries)...")
        tier_results["l1_cache"] = await measure_l1(medha, sample)
        print(f"    mean={tier_results['l1_cache']['mean_ms']:.2f}ms")

        results[str(size)] = tier_results
        await medha.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Medha search latency across tiers and collection sizes.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,1000,10000",
        help="Comma-separated collection sizes to benchmark (default: 100,1000,10000)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries per tier measurement (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print("Medha Latency Benchmark")
    print(f"  Sizes: {sizes}")
    print(f"  Queries per tier: {args.num_queries}")

    results = asyncio.run(benchmark(sizes, args.num_queries, args.seed))

    print(f"\n{'='*50}")
    print("Results Summary")
    print(f"{'='*50}")
    for size, tiers in results.items():
        print(f"\n  Collection size: {size}")
        for tier, stats in tiers.items():
            print(f"    {tier:20s}  mean={stats['mean_ms']:7.2f}ms  "
                  f"median={stats['median_ms']:7.2f}ms  "
                  f"p95={stats['p95_ms']:7.2f}ms")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
