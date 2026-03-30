"""Benchmark: score threshold sensitivity analysis.

Sweeps `score_threshold_exact` and `score_threshold_semantic` across a grid
and measures, for each combination:
    - Hit rate (fraction of queries that get any cache hit)
    - Precision (fraction of hits where the returned query is correct)
    - F1 score (harmonic mean of hit-rate and precision)
    - Hits by strategy (exact vs semantic)

A "correct" hit is one where the returned generated_query matches the
expected query stored for that question.

Usage:
    python experiments/threshold_sweep.py
    python experiments/threshold_sweep.py --exact-range 0.92,0.95,0.97,0.99 --semantic-range 0.80,0.85,0.90,0.95
    python experiments/threshold_sweep.py --output sweep_report.json
"""

import asyncio
import argparse
import json
import random
from statistics import mean

from medha import Medha, Settings, SearchStrategy
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset
from latency_benchmark import paraphrase, generate_unrelated


# ---------------------------------------------------------------------------
# Workload builder
# ---------------------------------------------------------------------------

def build_labeled_workload(
    dataset: list[dict],
    n_exact: int,
    n_paraphrase: int,
    n_novel: int,
) -> list[dict]:
    """Build a workload where each query has a known expected answer.

    Returns list of {"question": str, "expected_query": str | None}
    where None means no match is expected (novel/unrelated questions).
    """
    workload = []

    # Exact: same question, should hit with high-confidence exact match
    for entry in random.choices(dataset, k=n_exact):
        workload.append({"question": entry["question"], "expected_query": entry["query"]})

    # Paraphrase: reworded, should hit semantic tier (expected = original query)
    para_samples = random.choices(dataset, k=n_paraphrase)
    para_questions = paraphrase(para_samples)
    for q, entry in zip(para_questions, para_samples):
        workload.append({"question": q, "expected_query": entry["query"]})

    # Novel: should NOT match anything
    for q in generate_unrelated(n_novel):
        workload.append({"question": q, "expected_query": None})

    random.shuffle(workload)
    return workload


# ---------------------------------------------------------------------------
# Single sweep point
# ---------------------------------------------------------------------------

async def evaluate_thresholds(
    medha: Medha,
    workload: list[dict],
) -> dict:
    """Evaluate hit rate and precision for the current medha threshold settings."""
    hits = 0
    correct_hits = 0
    by_strategy: dict[str, int] = {}

    for item in workload:
        result = await medha.search(item["question"])
        strategy = result.strategy.value if result.strategy else "unknown"
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        expected = item["expected_query"]
        is_hit = result.strategy not in (SearchStrategy.NO_MATCH, SearchStrategy.ERROR)

        if is_hit:
            hits += 1
            # Precision: hit is "correct" if expected is not None and query matches
            if expected is not None and result.generated_query == expected:
                correct_hits += 1

    total = len(workload)
    n_positive = sum(1 for w in workload if w["expected_query"] is not None)

    hit_rate = hits / total if total > 0 else 0.0
    precision = correct_hits / hits if hits > 0 else 0.0
    # Recall: fraction of positive queries correctly returned
    recall = correct_hits / n_positive if n_positive > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "hit_rate":    round(hit_rate, 4),
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "hits":        hits,
        "correct_hits": correct_hits,
        "total":       total,
        "by_strategy": by_strategy,
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

async def run_sweep(
    exact_thresholds: list[float],
    semantic_thresholds: list[float],
    cache_size: int,
    n_exact: int,
    n_paraphrase: int,
    n_novel: int,
    seed: int | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)

    print(f"  Generating {cache_size} cache entries...")
    dataset = generate_dataset(cache_size, lang="sql")

    print(f"  Building workload (exact={n_exact}, paraphrase={n_paraphrase}, novel={n_novel})...")
    workload = build_labeled_workload(dataset, n_exact, n_paraphrase, n_novel)

    grid_results = []
    total_combos = len(exact_thresholds) * len(semantic_thresholds)
    combo_idx = 0

    for exact_t in exact_thresholds:
        for semantic_t in semantic_thresholds:
            # Skip invalid combinations (semantic must be < exact)
            if semantic_t >= exact_t:
                combo_idx += 1
                continue

            combo_idx += 1
            print(f"  [{combo_idx}/{total_combos}] exact={exact_t:.2f}, semantic={semantic_t:.2f}...",
                  end="", flush=True)

            embedder = FastEmbedAdapter()
            # backend_type="memory" — pure cosine similarity, no HNSW approximation.
            # Results are exact and reproducible across runs.
            settings = Settings(
                backend_type="memory",
                score_threshold_exact=exact_t,
                score_threshold_semantic=semantic_t,
            )
            medha = Medha("sweep_bench", embedder=embedder, settings=settings)
            await medha.start()

            for entry in dataset:
                await medha.store(entry["question"], entry["query"])

            # Clear L1 so each sweep point is evaluated from the vector store
            await medha.clear_caches()

            metrics = await evaluate_thresholds(medha, workload)
            await medha.close()

            grid_results.append({
                "exact_threshold":    exact_t,
                "semantic_threshold": semantic_t,
                **metrics,
            })

            print(f"  hit={metrics['hit_rate']:.3f}  prec={metrics['precision']:.3f}  f1={metrics['f1']:.3f}")

    # Find best combo by F1
    best = max(grid_results, key=lambda r: r["f1"]) if grid_results else None

    return {
        "cache_size": cache_size,
        "workload": {"n_exact": n_exact, "n_paraphrase": n_paraphrase, "n_novel": n_novel},
        "grid": grid_results,
        "best_by_f1": best,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    print(f"\n{'='*70}")
    print("Threshold Sensitivity Sweep Results")
    print(f"{'='*70}")
    print(f"  Cache size: {results['cache_size']}")
    w = results['workload']
    print(f"  Workload:   exact={w['n_exact']}, paraphrase={w['n_paraphrase']}, novel={w['n_novel']}")
    print()
    print(f"  {'exact':>6}  {'sem':>6}  {'hit_rate':>9}  {'precision':>9}  {'recall':>7}  {'f1':>6}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*6}")
    for r in sorted(results["grid"], key=lambda x: x["f1"], reverse=True)[:20]:
        print(f"  {r['exact_threshold']:6.2f}  {r['semantic_threshold']:6.2f}  "
              f"{r['hit_rate']:9.4f}  {r['precision']:9.4f}  "
              f"{r['recall']:7.4f}  {r['f1']:6.4f}")

    if results["best_by_f1"]:
        b = results["best_by_f1"]
        print(f"\n  Best configuration (by F1):")
        print(f"    exact={b['exact_threshold']:.2f}, semantic={b['semantic_threshold']:.2f}")
        print(f"    F1={b['f1']:.4f}, hit_rate={b['hit_rate']:.4f}, precision={b['precision']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Medha score thresholds to find optimal configuration",
    )
    parser.add_argument(
        "--exact-range", type=str, default="0.92,0.95,0.97,0.99",
        help="Comma-separated exact threshold values (default: 0.92,0.95,0.97,0.99)",
    )
    parser.add_argument(
        "--semantic-range", type=str, default="0.80,0.85,0.88,0.90,0.92,0.95",
        help="Comma-separated semantic threshold values (default: 0.80,0.85,0.88,0.90,0.92,0.95)",
    )
    parser.add_argument("--cache-size", type=int, default=200,
                        help="Cache entries to pre-populate (default: 200)")
    parser.add_argument("--n-exact", type=int, default=80,
                        help="Exact-repeat queries in workload (default: 80)")
    parser.add_argument("--n-paraphrase", type=int, default=80,
                        help="Paraphrase queries in workload (default: 80)")
    parser.add_argument("--n-novel", type=int, default=40,
                        help="Novel (miss) queries in workload (default: 40)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (prints to stdout if omitted)")
    args = parser.parse_args()

    exact_thresholds = [float(v.strip()) for v in args.exact_range.split(",")]
    semantic_thresholds = [float(v.strip()) for v in args.semantic_range.split(",")]

    print("Medha Threshold Sensitivity Sweep")
    print(f"  Exact thresholds:    {exact_thresholds}")
    print(f"  Semantic thresholds: {semantic_thresholds}")
    print(f"  Grid points: {len(exact_thresholds) * len(semantic_thresholds)}")

    results = asyncio.run(run_sweep(
        exact_thresholds, semantic_thresholds,
        args.cache_size, args.n_exact, args.n_paraphrase, args.n_novel,
        args.seed,
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
