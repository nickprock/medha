# Benchmarks

Performance and cost analysis scripts for Medha.
All scripts are in the [`experiments/`](https://github.com/ArchAI-Labs/medha/tree/main/experiments) folder and can be run locally.

```bash
git clone https://github.com/ArchAI-Labs/medha.git
cd medha
pip install "medha-archai[all]"
python experiments/<script_name>.py
```

---

## Latency

| Script | Description |
|---|---|
| [latency_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/latency_benchmark.py) | Per-tier search latency — L1 cache, exact vector match, semantic match, fuzzy |
| [backend_comparison_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/backend_comparison_benchmark.py) | Search latency comparison across all available backends |
| [l1_cache_backend_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/l1_cache_backend_benchmark.py) | `InMemoryL1Cache` vs `RedisL1Cache` — latency, hit rate, and throughput |
| [embedding_cache_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/embedding_cache_benchmark.py) | Persistent embedding cache warm-start vs cold-start latency |

---

## Throughput & Ingestion

| Script | Description |
|---|---|
| [batch_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/batch_benchmark.py) | Ingestion throughput — sequential `store()` vs `store_batch` vs `store_many` |
| [concurrency_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/concurrency_benchmark.py) | Embedding deduplication under concurrent load — verifies only one embedding call is made per in-flight question |

---

## Quality & Tuning

| Script | Description |
|---|---|
| [threshold_sweep.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/threshold_sweep.py) | Score threshold sensitivity analysis — sweeps `score_threshold_exact` and `score_threshold_semantic` and measures hit rate vs precision |
| [gliner_vs_regex_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/gliner_vs_regex_benchmark.py) | Parameter extraction comparison — regex vs GLiNER vs spaCy |

---

## Cost & Observability

| Script | Description |
|---|---|
| [cost_benchmark.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/cost_benchmark.py) | LLM-only cost vs Medha-assisted cost — simulates a realistic workload and estimates API savings |
| [observability_demo.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/observability_demo.py) | Live `CacheStats` snapshots — hit rate, miss rate, avg latency, p95, per-strategy breakdown |

---

## Dataset Generation

| Script | Description |
|---|---|
| [dataset_generation.py](https://github.com/ArchAI-Labs/medha/blob/main/experiments/dataset_generation.py) | Generate synthetic question-query pairs for benchmarking (`--count`, `--lang`, `--output`) |
