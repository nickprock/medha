"""Benchmark: GliNER vs regex for template parameter extraction.

Compares the three parameter extraction backends available in Medha's
ParameterExtractor cascade:
    1. Regex patterns (fast, deterministic, requires hand-crafted patterns)
    2. GliNER zero-shot NER (slow, flexible, uses param names as labels)
    3. Heuristic fallback (numbers + capitalized words, no dependencies)

Metrics per backend:
    - Extraction recall  : fraction of parameters correctly extracted
    - Exact-match rate   : fraction of queries where ALL params are extracted
    - Latency            : mean extraction time per question (ms)
    - Requires install   : whether an optional package is needed

GliNER requires: pip install "medha[gliner]"
If not installed, the GliNER scenario is skipped gracefully.

Usage:
    python experiments/gliner_vs_regex_benchmark.py
    python experiments/gliner_vs_regex_benchmark.py --num-questions 200 --output gliner_report.json
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, field, asdict
from statistics import mean, median

from medha.utils.nlp import ParameterExtractor
from medha.types import QueryTemplate


# ---------------------------------------------------------------------------
# Template fixtures with known ground-truth parameters
# ---------------------------------------------------------------------------

# Each fixture: (template, list of (question, expected_params_dict))
FIXTURES: list[tuple[QueryTemplate, list[tuple[str, dict]]]] = [
    # --- Numeric + entity type (regex-friendly) ---
    (
        QueryTemplate(
            intent="top_n_entity",
            template_text="Show the top {count} {entity}",
            query_template="SELECT * FROM {entity} LIMIT {count}",
            parameters=["count", "entity"],
            parameter_patterns={
                "count":  r"\b(\d+)\b",
                "entity": r"top \d+ (\w+)",
            },
        ),
        [
            ("Show the top 10 products", {"count": "10", "entity": "products"}),
            ("Show the top 5 employees", {"count": "5", "entity": "employees"}),
            ("Show the top 20 orders",   {"count": "20", "entity": "orders"}),
            ("Show the top 3 users",     {"count": "3", "entity": "users"}),
            ("Show the top 100 items",   {"count": "100", "entity": "items"}),
        ],
    ),
    # --- Proper name (NER-friendly) ---
    (
        QueryTemplate(
            intent="find_person_friends",
            template_text="Find {person}'s friends",
            query_template="MATCH (p:Person {{name:'{person}'}})-[:FRIEND]-(f) RETURN f.name",
            parameters=["person"],
            parameter_patterns={},  # No regex — intentionally left empty
        ),
        [
            ("Find Alice's friends",   {"person": "Alice"}),
            ("Find Bob's friends",     {"person": "Bob"}),
            ("Find Charlie's friends", {"person": "Charlie"}),
            ("Find Diana's friends",   {"person": "Diana"}),
            ("Find Eve's friends",     {"person": "Eve"}),
        ],
    ),
    # --- Organisation + department (domain-specific, GliNER advantage) ---
    (
        QueryTemplate(
            intent="org_department_employees",
            template_text="Who works in {department} at {company}?",
            query_template=(
                "MATCH (p:Person)-[:WORKS_IN]->(d:Department {{name:'{department}'}})"
                "-[:PART_OF]->(c:Company {{name:'{company}'}}) RETURN p.name"
            ),
            parameters=["department", "company"],
            parameter_patterns={
                "company":    r"at (\w[\w ]+?)(?:\?|$)",
                "department": r"in (\w[\w ]+?) at",
            },
        ),
        [
            ("Who works in Engineering at Acme?",   {"department": "Engineering", "company": "Acme"}),
            ("Who works in Marketing at Globex?",    {"department": "Marketing",   "company": "Globex"}),
            ("Who works in Sales at Initech?",       {"department": "Sales",       "company": "Initech"}),
            ("Who works in Research at Umbrella?",   {"department": "Research",    "company": "Umbrella"}),
            ("Who works in Finance at Stark Industries?", {"department": "Finance", "company": "Stark Industries"}),
        ],
    ),
    # --- Count query (regex-friendly) ---
    (
        QueryTemplate(
            intent="count_entity",
            template_text="How many {entity} are there?",
            query_template="SELECT COUNT(*) FROM {entity}",
            parameters=["entity"],
            parameter_patterns={
                "entity": r"How many (\w+) are",
            },
        ),
        [
            ("How many users are there?",    {"entity": "users"}),
            ("How many orders are there?",   {"entity": "orders"}),
            ("How many products are there?", {"entity": "products"}),
            ("How many employees are there?",{"entity": "employees"}),
            ("How many items are there?",    {"entity": "items"}),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    backend: str
    available: bool
    per_template: dict = field(default_factory=dict)  # intent -> metrics
    overall_recall: float = 0.0
    overall_exact_match_rate: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0


def _evaluate_extractor(
    extractor: ParameterExtractor,
    backend_name: str,
    fixtures: list | None = None,
) -> ExtractionResult:
    active_fixtures = fixtures if fixtures is not None else FIXTURES
    result = ExtractionResult(backend=backend_name, available=True)
    all_latencies: list[float] = []
    all_param_hits = 0
    all_param_total = 0
    all_exact_hits = 0
    all_exact_total = 0

    for template, cases in active_fixtures:
        param_hits = 0
        param_total = 0
        exact_hits = 0

        for question, expected in cases:
            t_start = time.perf_counter()
            try:
                extracted = extractor.extract(question, template)
            except Exception:
                extracted = {}
            latency_ms = (time.perf_counter() - t_start) * 1000
            all_latencies.append(latency_ms)

            # Per-parameter recall
            for param, expected_val in expected.items():
                param_total += 1
                if extracted.get(param, "").lower() == expected_val.lower():
                    param_hits += 1

            # Exact match: all params extracted correctly
            all_correct = all(
                extracted.get(p, "").lower() == v.lower()
                for p, v in expected.items()
            )
            if all_correct:
                exact_hits += 1

        all_param_hits += param_hits
        all_param_total += param_total
        all_exact_hits += exact_hits
        all_exact_total += len(cases)

        result.per_template[template.intent] = {
            "recall":          round(param_hits / param_total, 4) if param_total else 0.0,
            "exact_match_rate": round(exact_hits / len(cases), 4) if cases else 0.0,
        }

    result.overall_recall = round(all_param_hits / all_param_total, 4) if all_param_total else 0.0
    result.overall_exact_match_rate = round(all_exact_hits / all_exact_total, 4) if all_exact_total else 0.0
    result.mean_latency_ms = round(mean(all_latencies), 3) if all_latencies else 0.0
    result.median_latency_ms = round(median(all_latencies), 3) if all_latencies else 0.0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(num_questions: int, seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)

    results = {}

    # --- Regex only ---
    print("  Running: regex-only...")
    regex_extractor = ParameterExtractor(use_gliner=False, use_spacy=False)
    regex_result = _evaluate_extractor(regex_extractor, "regex")
    results["regex"] = asdict(regex_result)
    print(f"    recall={regex_result.overall_recall:.4f}  "
          f"exact={regex_result.overall_exact_match_rate:.4f}  "
          f"latency={regex_result.mean_latency_ms:.3f}ms")

    # --- Heuristic fallback only (no regex patterns) ---
    print("  Running: heuristic-only (no regex, no NER)...")
    # Strip all patterns to force the heuristic fallback path
    stripped_fixtures = [
        (template.model_copy(update={"parameter_patterns": {}}), cases)
        for template, cases in FIXTURES
    ]
    heuristic_extractor = ParameterExtractor(use_gliner=False, use_spacy=False)
    heuristic_result = _evaluate_extractor(heuristic_extractor, "heuristic", stripped_fixtures)
    results["heuristic"] = asdict(heuristic_result)
    print(f"    recall={heuristic_result.overall_recall:.4f}  "
          f"exact={heuristic_result.overall_exact_match_rate:.4f}  "
          f"latency={heuristic_result.mean_latency_ms:.3f}ms")

    # --- GliNER ---
    print("  Running: gliner...")
    try:
        gliner_extractor = ParameterExtractor(use_gliner=True, use_spacy=False)
        if not gliner_extractor._gliner_available:
            raise ImportError("GliNER not available")
        gliner_result = _evaluate_extractor(gliner_extractor, "gliner")
        results["gliner"] = asdict(gliner_result)
        print(f"    recall={gliner_result.overall_recall:.4f}  "
              f"exact={gliner_result.overall_exact_match_rate:.4f}  "
              f"latency={gliner_result.mean_latency_ms:.3f}ms")
    except (ImportError, AttributeError) as exc:
        print(f"    skipped ({exc})")
        results["gliner"] = {"available": False, "reason": str(exc)}

    # --- GliNER + Regex cascade ---
    print("  Running: regex + gliner cascade...")
    try:
        cascade_extractor = ParameterExtractor(use_gliner=True, use_spacy=False)
        if not cascade_extractor._gliner_available:
            raise ImportError("GliNER not available")
        cascade_result = _evaluate_extractor(cascade_extractor, "regex+gliner")
        results["regex_gliner_cascade"] = asdict(cascade_result)
        print(f"    recall={cascade_result.overall_recall:.4f}  "
              f"exact={cascade_result.overall_exact_match_rate:.4f}  "
              f"latency={cascade_result.mean_latency_ms:.3f}ms")
    except (ImportError, AttributeError) as exc:
        print(f"    skipped ({exc})")
        results["regex_gliner_cascade"] = {"available": False, "reason": str(exc)}

    return results


def _print_summary(results: dict) -> None:
    print(f"\n{'='*70}")
    print("GliNER vs Regex Parameter Extraction Benchmark")
    print(f"{'='*70}")
    print(f"\n  {'Backend':25s}  {'Recall':>7}  {'Exact%':>7}  {'mean_ms':>9}  {'available':>9}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")

    for backend_key in ("regex", "heuristic", "gliner", "regex_gliner_cascade"):
        if backend_key not in results:
            continue
        r = results[backend_key]
        if not r.get("available", True):
            print(f"  {backend_key:25s}  {'n/a':>7}  {'n/a':>7}  {'n/a':>9}  {'no':>9}")
            continue
        print(f"  {r['backend']:25s}  "
              f"{r['overall_recall']:7.4f}  "
              f"{r['overall_exact_match_rate']:7.4f}  "
              f"{r['mean_latency_ms']:9.3f}  "
              f"{'yes':>9}")

    # Per-template breakdown
    print(f"\n  Per-template recall (regex vs gliner):")
    regex_per = results.get("regex", {}).get("per_template", {})
    gliner_per = results.get("gliner", {}).get("per_template", {}) if results.get("gliner", {}).get("available", False) else {}
    for intent in regex_per:
        rx = regex_per[intent]["recall"]
        gl = gliner_per.get(intent, {}).get("recall", float("nan"))
        gl_str = f"{gl:.4f}" if gl == gl else "n/a"
        print(f"    {intent:30s}  regex={rx:.4f}  gliner={gl_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GliNER vs regex for template parameter extraction",
    )
    parser.add_argument("--num-questions", type=int, default=100,
                        help="Number of questions per template (default: 100; "
                             "currently uses fixed fixture set, this arg is reserved)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (prints to stdout if omitted)")
    args = parser.parse_args()

    print("Medha GliNER vs Regex Parameter Extraction Benchmark")

    results = run_benchmark(args.num_questions, args.seed)
    _print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
