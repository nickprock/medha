"""Benchmark: search latency comparison across all Medha backends.

Runs the same workload on every available backend and prints a comparison table.
Backends with missing packages or unreachable services are skipped automatically.

Usage:
    python experiments/backend_comparison_benchmark.py
    python experiments/backend_comparison_benchmark.py --size 200 --queries 100
    python experiments/backend_comparison_benchmark.py --backends memory,chroma,lancedb
    python experiments/backend_comparison_benchmark.py --size 500 --output comparison.json

Output:
    Backend         Status   Store(eps)  Search mean  p50    p95    p99
    memory          ok       2340.1      1.23ms       1.10   2.45   3.80
    chroma          ok       1890.4      1.45ms       1.30   2.80   4.10
    qdrant          ok       1120.3      1.67ms       1.50   3.10   4.90
    elasticsearch   skipped  (service not reachable at localhost:9200)
    redis           skipped  (pip install medha-archai[redis] required)
"""

import asyncio
import argparse
import importlib.util
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).parent))

from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

from dataset_generation import generate_dataset


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

def _tmp_lancedb_uri() -> str:
    return tempfile.mkdtemp(prefix="medha_cmp_lancedb_")


BACKENDS_CONFIG: dict[str, dict] = {
    "memory": {
        "settings_kwargs": {"backend_type": "memory"},
        "requires": None,
        "probe": None,
    },
    "qdrant": {
        "settings_kwargs": {"backend_type": "qdrant", "qdrant_mode": "memory"},
        "requires": "qdrant_client",
        "probe": None,
    },
    "pgvector": {
        "settings_kwargs": {"backend_type": "pgvector"},
        "requires": "asyncpg",
        "probe": ("localhost", 5432),
    },
    "elasticsearch": {
        "settings_kwargs": {"backend_type": "elasticsearch"},
        "requires": "elasticsearch",
        "probe": ("localhost", 9200),
    },
    "vectorchord": {
        "settings_kwargs": {"backend_type": "vectorchord"},
        "requires": "asyncpg",
        "probe": ("localhost", 5432),
    },
    "chroma": {
        "settings_kwargs": {"backend_type": "chroma", "chroma_mode": "ephemeral"},
        "requires": "chromadb",
        "probe": None,
    },
    "weaviate": {
        "settings_kwargs": {"backend_type": "weaviate"},
        "requires": "weaviate",
        "probe": ("localhost", 8080),
    },
    "redis": {
        "settings_kwargs": {"backend_type": "redis"},
        "requires": "redis",
        "probe": ("localhost", 6379),
    },
    "azure-search": {
        "settings_kwargs": {"backend_type": "azure-search"},
        "requires": "azure.search.documents",
        "probe": None,
    },
    "lancedb": {
        "settings_kwargs": {"backend_type": "lancedb"},
        "requires": "lancedb",
        "probe": None,
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _has_package(name: str | None) -> bool:
    if name is None:
        return True
    return importlib.util.find_spec(name) is not None


async def _is_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def select_backends(requested: list[str]) -> list[tuple[str, dict]]:
    """Filter requested backends by installed packages and TCP reachability."""
    selected = []
    for name in requested:
        if name not in BACKENDS_CONFIG:
            print(f"  [skip] {name}: unknown backend")
            continue

        cfg = BACKENDS_CONFIG[name]
        pkg = cfg["requires"]
        probe = cfg["probe"]

        if not _has_package(pkg):
            print(f"  [skip] {name}: pip install medha-archai[{name}] required")
            continue

        if probe is not None:
            host, port = probe
            if not await _is_reachable(host, port):
                print(f"  [skip] {name}: service not reachable at {host}:{port}")
                continue

        selected.append((name, cfg))

    return selected


# ---------------------------------------------------------------------------
# Per-backend scenario
# ---------------------------------------------------------------------------

async def run_backend_scenario(
    name: str,
    settings_kwargs: dict,
    dataset: list[dict],
    queries: list[str],
) -> dict:
    """Run store + search workload for one backend and return metrics."""
    # Resolve lancedb URI lazily so each call gets its own tmpdir
    kwargs = dict(settings_kwargs)
    if kwargs.get("backend_type") == "lancedb" and "lancedb_uri" not in kwargs:
        kwargs["lancedb_uri"] = _tmp_lancedb_uri()

    embedder = FastEmbedAdapter()
    settings = Settings(**kwargs)
    medha = Medha(f"cmp_{name}", embedder=embedder, settings=settings)

    result: dict = {"backend": name, "status": "ok"}

    try:
        try:
            await medha.start()
        except Exception as exc:
            result["status"] = "error_start"
            result["error"] = str(exc)
            return result

        # Store
        entries = [{"question": d["question"], "generated_query": d["query"]} for d in dataset]
        t0 = time.perf_counter()
        await medha.store_many(entries)
        store_s = time.perf_counter() - t0
        result["store_eps"] = round(len(entries) / store_s, 1) if store_s > 0 else 0.0

        # Search
        latencies_ms: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            await medha.search(q)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        if latencies_ms:
            sorted_lats = sorted(latencies_ms)
            n = len(sorted_lats)
            result["search_mean_ms"] = round(mean(latencies_ms), 2)
            result["search_p50_ms"] = round(median(latencies_ms), 2)
            result["search_p95_ms"] = round(sorted_lats[min(int(n * 0.95), n - 1)], 2)
            result["search_p99_ms"] = round(sorted_lats[min(int(n * 0.99), n - 1)], 2)
        else:
            result["search_mean_ms"] = 0.0
            result["search_p50_ms"] = 0.0
            result["search_p95_ms"] = 0.0
            result["search_p99_ms"] = 0.0

    finally:
        try:
            await medha.close()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_table(results: list[dict]) -> None:
    header = f"  {'Backend':<16} {'Status':<10} {'Store(eps)':>11}  {'Mean':>8}  {'p50':>7}  {'p95':>7}  {'p99':>7}"
    print(f"\n{header}")
    print(f"  {'-'*74}")
    for r in results:
        status = r.get("status", "?")
        if status == "ok":
            row = (
                f"  {r['backend']:<16} {status:<10} "
                f"{r.get('store_eps', 0):>11.1f}  "
                f"{r.get('search_mean_ms', 0):>7.2f}ms  "
                f"{r.get('search_p50_ms', 0):>6.2f}ms  "
                f"{r.get('search_p95_ms', 0):>6.2f}ms  "
                f"{r.get('search_p99_ms', 0):>6.2f}ms"
            )
        elif status == "error_start":
            row = f"  {r['backend']:<16} {'error':<10} {r.get('error', '')}"
        else:
            row = f"  {r['backend']:<16} {'skipped':<10}"
        print(row)


async def main_async(
    size: int,
    num_queries: int,
    requested_backends: list[str],
    seed: int | None,
) -> list[dict]:
    if seed is not None:
        random.seed(seed)

    print("Medha Backend Comparison Benchmark")
    print(f"  Dataset: {size} entries  |  Queries: {num_queries}")
    print(f"\nChecking backend availability...")

    selected = await select_backends(requested_backends)

    if not selected:
        print("No backends available. Exiting.")
        return []

    dataset = generate_dataset(size, lang="sql")
    query_pool = [d["question"] for d in dataset]
    queries = random.choices(query_pool, k=num_queries)

    results = []
    for name, cfg in selected:
        print(f"  Running {name}...")
        r = await run_backend_scenario(name, cfg["settings_kwargs"], dataset, queries)
        results.append(r)

    _print_table(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare search latency across Medha backends on the same workload.",
    )
    parser.add_argument(
        "--size", type=int, default=200,
        help="Number of entries to store in each backend (default: 200)",
    )
    parser.add_argument(
        "--queries", type=int, default=100,
        help="Number of search queries to run per backend (default: 100)",
    )
    parser.add_argument(
        "--backends", type=str,
        default="memory,chroma,qdrant,pgvector,elasticsearch,vectorchord,weaviate,redis,azure-search,lancedb",
        help="Comma-separated list of backends to benchmark (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to this JSON file (optional)",
    )
    args = parser.parse_args()

    requested = [b.strip() for b in args.backends.split(",")]

    results = asyncio.run(main_async(args.size, args.queries, requested, args.seed))

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    elif results:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
