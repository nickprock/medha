"""Benchmark: LLM-only cost vs Medha-assisted cost.

Simulates a workload of N questions against a pre-populated Medha cache and
compares the estimated cost of always calling an LLM vs using Medha to serve
cache hits and only calling the LLM on misses.

No actual LLM API calls are made --- costs are estimated from token counts.

Usage:
    python experiments/cost_benchmark.py --num-queries 1000 --cache-size 500
    python experiments/cost_benchmark.py --num-queries 5000 --cache-size 1000 --output cost_report.json
"""

import asyncio
import argparse
import json
import random
import time
from dataclasses import dataclass, field, asdict
from statistics import mean

from medha import Medha, Settings, SearchStrategy
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset
from latency_benchmark import paraphrase, generate_unrelated


# ---------------------------------------------------------------------------
# LLM pricing (per 1M tokens)
# ---------------------------------------------------------------------------

LLM_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}

# Average tokens per text-to-query interaction (estimated)
AVG_PROMPT_TOKENS = 250     # system prompt + user question + schema context
AVG_COMPLETION_TOKENS = 60  # generated SQL/Cypher/MQL query


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostReport:
    """Aggregated cost comparison report."""
    num_queries: int = 0
    cache_size: int = 0

    # Medha results
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    hits_by_strategy: dict = field(default_factory=dict)

    # LLM-only cost (every query triggers an LLM call)
    llm_only_calls: int = 0
    llm_only_input_tokens: int = 0
    llm_only_output_tokens: int = 0
    llm_only_cost: dict = field(default_factory=dict)  # per model

    # Medha-assisted cost (only misses trigger an LLM call)
    medha_assisted_calls: int = 0
    medha_assisted_input_tokens: int = 0
    medha_assisted_output_tokens: int = 0
    medha_assisted_cost: dict = field(default_factory=dict)  # per model

    # Savings
    calls_saved: int = 0
    calls_saved_pct: float = 0.0
    cost_savings: dict = field(default_factory=dict)  # per model: {absolute, percentage}

    # Timing
    total_time_s: float = 0.0
    avg_search_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Workload builder
# ---------------------------------------------------------------------------

def build_workload(dataset: list[dict], num_queries: int) -> list[str]:
    """Build a realistic workload from the dataset.

    Distribution:
        - 40% exact repeats (same question from cache)
        - 30% paraphrases (slightly reworded, should get semantic hits)
        - 30% novel questions (cache misses)
    """
    n_exact = int(num_queries * 0.40)
    n_paraphrase = int(num_queries * 0.30)
    n_novel = num_queries - n_exact - n_paraphrase  # remainder goes to novel

    workload = []

    # 40% exact repeats — sample with replacement from stored questions
    exact_samples = random.choices(dataset, k=n_exact)
    workload.extend(entry["question"] for entry in exact_samples)

    # 30% paraphrases — paraphrase random samples from the dataset
    para_samples = random.choices(dataset, k=n_paraphrase)
    workload.extend(paraphrase(para_samples))

    # 30% novel — unrelated questions that should miss the cache
    workload.extend(generate_unrelated(n_novel))

    # Shuffle to simulate a realistic interleaved workload
    random.shuffle(workload)

    return workload


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    num_queries: int,
    cache_size: int,
    llm_models: list[str],
    seed: int | None = None,
) -> CostReport:
    """Run the cost benchmark."""
    if seed is not None:
        random.seed(seed)

    report = CostReport(num_queries=num_queries, cache_size=cache_size)

    # --- Setup Medha ---
    embedder = FastEmbedAdapter()
    settings = Settings(qdrant_mode="memory")
    medha = Medha("cost_bench", embedder=embedder, settings=settings)
    await medha.start()

    # --- Populate cache ---
    print(f"Generating {cache_size} cache entries...")
    dataset = generate_dataset(cache_size, lang="sql")

    print(f"Storing {cache_size} entries in Medha...")
    for entry in dataset:
        await medha.store(entry["question"], entry["query"])

    # --- Build workload ---
    print(f"Building workload of {num_queries} queries "
          f"(40% exact, 30% paraphrase, 30% novel)...")
    workload = build_workload(dataset, num_queries)

    # --- Run workload through Medha ---
    print(f"Running workload...")
    search_times = []
    hits_by_strategy: dict[str, int] = {}

    for i, question in enumerate(workload):
        start = time.perf_counter()
        hit = await medha.search(question)
        elapsed = (time.perf_counter() - start) * 1000
        search_times.append(elapsed)

        strategy = hit.strategy.value
        hits_by_strategy[strategy] = hits_by_strategy.get(strategy, 0) + 1

        if hit.strategy == SearchStrategy.NO_MATCH:
            report.cache_misses += 1
        else:
            report.cache_hits += 1

        # Progress indicator
        if (i + 1) % 200 == 0 or (i + 1) == len(workload):
            print(f"  {i + 1}/{len(workload)} queries processed "
                  f"(hits: {report.cache_hits}, misses: {report.cache_misses})")

    await medha.close()

    # --- Compute metrics ---
    report.hit_rate = report.cache_hits / num_queries if num_queries > 0 else 0.0
    report.hits_by_strategy = hits_by_strategy
    report.total_time_s = round(sum(search_times) / 1000, 2)
    report.avg_search_time_ms = round(mean(search_times), 2) if search_times else 0.0

    # LLM-only: every query costs tokens
    report.llm_only_calls = num_queries
    report.llm_only_input_tokens = num_queries * AVG_PROMPT_TOKENS
    report.llm_only_output_tokens = num_queries * AVG_COMPLETION_TOKENS

    # Medha-assisted: only misses cost tokens
    report.medha_assisted_calls = report.cache_misses
    report.medha_assisted_input_tokens = report.cache_misses * AVG_PROMPT_TOKENS
    report.medha_assisted_output_tokens = report.cache_misses * AVG_COMPLETION_TOKENS

    report.calls_saved = report.cache_hits
    report.calls_saved_pct = round(report.hit_rate * 100, 2)

    # Cost per model
    for model, pricing in LLM_PRICING.items():
        if model not in llm_models:
            continue

        llm_cost = (
            report.llm_only_input_tokens * pricing["input"]
            + report.llm_only_output_tokens * pricing["output"]
        ) / 1_000_000

        medha_cost = (
            report.medha_assisted_input_tokens * pricing["input"]
            + report.medha_assisted_output_tokens * pricing["output"]
        ) / 1_000_000

        report.llm_only_cost[model] = round(llm_cost, 6)
        report.medha_assisted_cost[model] = round(medha_cost, 6)
        report.cost_savings[model] = {
            "absolute_usd": round(llm_cost - medha_cost, 6),
            "percentage": round((1 - medha_cost / llm_cost) * 100, 2) if llm_cost > 0 else 0.0,
        }

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(report: CostReport) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*60}")
    print("Cost Benchmark Results")
    print(f"{'='*60}")
    print(f"  Workload:       {report.num_queries} queries")
    print(f"  Cache size:     {report.cache_size} entries")
    print(f"  Cache hits:     {report.cache_hits} ({report.calls_saved_pct:.1f}%)")
    print(f"  Cache misses:   {report.cache_misses}")
    print(f"  Avg search:     {report.avg_search_time_ms:.2f}ms")
    print(f"  Total time:     {report.total_time_s:.2f}s")

    print(f"\n  Strategy breakdown:")
    for strategy, count in sorted(report.hits_by_strategy.items()):
        pct = count / report.num_queries * 100
        print(f"    {strategy:20s}  {count:6d}  ({pct:5.1f}%)")

    if report.cost_savings:
        print(f"\n  Cost comparison (per model):")
        for model in report.llm_only_cost:
            llm = report.llm_only_cost[model]
            medha = report.medha_assisted_cost[model]
            saving = report.cost_savings[model]
            print(f"    {model}:")
            print(f"      LLM-only:       ${llm:.6f}")
            print(f"      Medha-assisted: ${medha:.6f}")
            print(f"      Saved:          ${saving['absolute_usd']:.6f} "
                  f"({saving['percentage']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Medha vs LLM-only cost benchmark",
    )
    parser.add_argument(
        "--num-queries", type=int, default=1000,
        help="Total number of queries in the workload (default: 1000)",
    )
    parser.add_argument(
        "--cache-size", type=int, default=500,
        help="Number of entries to pre-populate in Medha (default: 500)",
    )
    parser.add_argument(
        "--models", type=str, default="gpt-4o-mini,gpt-4o",
        help="Comma-separated LLM models to compare pricing (default: gpt-4o-mini,gpt-4o)",
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

    models = [m.strip() for m in args.models.split(",")]

    print("Medha Cost Benchmark")
    print(f"  Queries: {args.num_queries}, Cache: {args.cache_size}, Models: {models}")

    report = asyncio.run(run_benchmark(
        args.num_queries, args.cache_size, models, args.seed,
    ))

    _print_summary(report)

    result = asdict(report)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
