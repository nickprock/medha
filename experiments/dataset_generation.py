"""Generate synthetic question-query pairs for Medha benchmarking.

Usage:
    python experiments/dataset_generation.py --count 1000 --output dataset.json --lang sql
    python experiments/dataset_generation.py --count 500 --lang cypher --output cypher_dataset.json
"""

import argparse
import json
import random


# ---------------------------------------------------------------------------
# SQL patterns
# ---------------------------------------------------------------------------

SQL_PATTERNS = [
    ("How many {entity} are there?", "SELECT COUNT(*) FROM {entity}"),
    ("Show the top {n} {entity}", "SELECT * FROM {entity} LIMIT {n}"),
    (
        "List all {entity} where {col} = '{val}'",
        "SELECT * FROM {entity} WHERE {col} = '{val}'",
    ),
    (
        "What is the average {col} of {entity}?",
        "SELECT AVG({col}) FROM {entity}",
    ),
    (
        "Find {entity} ordered by {col}",
        "SELECT * FROM {entity} ORDER BY {col}",
    ),
]

# ---------------------------------------------------------------------------
# Cypher patterns
# ---------------------------------------------------------------------------

CYPHER_PATTERNS = [
    (
        "How many {label} are in the database?",
        "MATCH (n:{label}) RETURN COUNT(n)",
    ),
    (
        "Find {name}'s friends",
        "MATCH (p:Person {{name:'{name}'}})-[:FRIEND]-(f) RETURN f.name",
    ),
    (
        "Who works at {company}?",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company {{name:'{company}'}}) RETURN p.name",
    ),
    (
        "List all {label} with {prop} = '{val}'",
        "MATCH (n:{label} {{{prop}:'{val}'}}) RETURN n",
    ),
    (
        "Find the top {n} {label} ordered by {prop}",
        "MATCH (n:{label}) RETURN n ORDER BY n.{prop} DESC LIMIT {n}",
    ),
]

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

ENTITIES = ["users", "products", "orders", "employees", "departments", "projects"]
COLUMNS = ["name", "salary", "price", "date", "status", "category"]
VALUES = ["active", "pending", "Engineering", "Marketing", "Sales"]

LABELS = ["Person", "Company", "Project", "Department", "Team"]
NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve",
    "Frank", "Grace", "Hank", "Iris", "John",
]
COMPANIES = ["Acme", "Globex", "Initech", "Umbrella", "Stark Industries"]
PROPS = ["name", "age", "status", "level", "score"]
PROP_VALUES = ["active", "senior", "completed", "open", "archived"]


def _fill_sql(question_tpl: str, query_tpl: str) -> dict:
    """Fill a single SQL pattern with random vocabulary."""
    entity = random.choice(ENTITIES)
    col = random.choice(COLUMNS)
    val = random.choice(VALUES)
    n = str(random.randint(1, 20))

    question = question_tpl.format(entity=entity, col=col, val=val, n=n)
    query = query_tpl.format(entity=entity, col=col, val=val, n=n)
    return {"question": question, "query": query}


def _fill_cypher(question_tpl: str, query_tpl: str) -> dict:
    """Fill a single Cypher pattern with random vocabulary."""
    label = random.choice(LABELS)
    name = random.choice(NAMES)
    company = random.choice(COMPANIES)
    prop = random.choice(PROPS)
    val = random.choice(PROP_VALUES)
    n = str(random.randint(1, 20))

    question = question_tpl.format(
        label=label, name=name, company=company, prop=prop, val=val, n=n,
    )
    query = query_tpl.format(
        label=label, name=name, company=company, prop=prop, val=val, n=n,
    )
    return {"question": question, "query": query}


def generate_dataset(count: int, lang: str = "sql") -> list:
    """Generate ``count`` question-query pairs.

    Each pair is a dict with ``"question"`` and ``"query"`` keys.
    Patterns are chosen uniformly at random and filled with random vocabulary.
    """
    if lang == "sql":
        patterns = SQL_PATTERNS
        filler = _fill_sql
    elif lang == "cypher":
        patterns = CYPHER_PATTERNS
        filler = _fill_cypher
    else:
        raise ValueError(f"Unsupported language: {lang}")

    dataset = []
    for _ in range(count):
        question_tpl, query_tpl = random.choice(patterns)
        dataset.append(filler(question_tpl, query_tpl))

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic question-query pairs for Medha benchmarking.",
    )
    parser.add_argument("--count", type=int, default=1000, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default="dataset.json", help="Output JSON path")
    parser.add_argument("--lang", choices=["sql", "cypher"], default="sql", help="Query language")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    dataset = generate_dataset(args.count, args.lang)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
