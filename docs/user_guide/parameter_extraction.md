# Parameter Extraction

Template matching requires extracting parameter values (e.g. `{department}`, `{person}`) from the user's question before filling them into a query template. `ParameterExtractor` applies a **cascading strategy** — each step fills in only the parameters the previous step missed:

1. **Regex** — patterns defined in `template.parameter_patterns` (fastest, most precise)
2. **GLiNER** — zero-shot NER, uses `template.parameters` directly as entity labels
3. **spaCy** — pre-trained NER with a fixed label set mapped to parameter names
4. **Heuristics** — numbers and capitalized words as last resort

| Backend | Approach | Labels | Model size |
|---|---|---|---|
| **spaCy** | Pre-trained | Fixed (`PERSON`, `ORG`, `CARDINAL`, …) mapped to param names | ~15 MB |
| **GLiNER** | Zero-shot transformer | Arbitrary — uses `template.parameters` directly as labels | ~500 MB |

**Requirements:**

```bash
pip install "medha-archai[nlp]"       # spaCy
python -m spacy download en_core_web_sm

pip install "medha-archai[gliner]"    # GLiNER
```

---

## Defining Templates

```python
from medha.types import QueryTemplate
from medha.utils.nlp import ParameterExtractor

# Numeric + entity type — regex-friendly
top_n_template = QueryTemplate(
    intent="top_n",
    template_text="Show top {count} {entity}",
    query_template="SELECT * FROM {entity} LIMIT {count}",
    parameters=["count", "entity"],
    parameter_patterns={
        "count":  r"\b(\d+)\b",
        "entity": r"\b(users|products|orders|employees)\b",
    },
)

# Proper name — NER-friendly
person_template = QueryTemplate(
    intent="find_person",
    template_text="Find issues assigned to {person}",
    query_template="SELECT * FROM issues WHERE assignee = '{person}'",
    parameters=["person"],
    # No regex: relies on NER or heuristics
)

# Two domain-specific entities — ideal for GLiNER
org_project_template = QueryTemplate(
    intent="org_project_issues",
    template_text="Show open issues for {org} on project {project}",
    query_template="SELECT * FROM issues WHERE org='{org}' AND project='{project}' AND status='open'",
    parameters=["org", "project"],
    # No regex: org and project names are arbitrary strings
)
```

---

## Regex + Heuristics Only

No NER backend loaded. Uses `parameter_patterns` first, then falls back to numbers and capitalized words.

```python
ext = ParameterExtractor(use_spacy=False, use_gliner=False)

params = ext.extract("Show top 10 products", top_n_template)
# {"count": "10", "entity": "products"}

query = ext.render_query(top_n_template, params)
# SELECT * FROM products LIMIT 10
```

---

## spaCy

spaCy recognizes standard entity types (`PERSON`, `ORG`, `CARDINAL`) and maps them to parameter names via an internal mapping table.

```python
ext = ParameterExtractor(use_spacy=True, use_gliner=False)
print(ext.spacy_available)  # True if en_core_web_sm is installed

params = ext.extract("find issues assigned to Alice", person_template)
# {"person": "Alice"}
```

Inspect raw entities for debugging:

```python
entities = ext.extract_entities("find issues assigned to Alice Johnson")
# {"PERSON": "Alice Johnson", ...}
```

> If spaCy or `en_core_web_sm` is not installed, the extractor falls back gracefully to the next tier.

---

## GLiNER

GLiNER's key advantage: it receives `template.parameters` directly as entity labels — no mapping table needed. It determines what to extract from the label name itself.

```python
ext = ParameterExtractor(use_spacy=False, use_gliner=True)
print(ext.gliner_available)  # True if gliner is installed

# GLiNER is called as: model.predict_entities(question, labels=["org", "project"])
params = ext.extract(
    "show open issues for Acme Corp on project Apollo",
    org_project_template,
)
# {"org": "Acme Corp", "project": "Apollo"}

query = ext.render_query(org_project_template, params)
# SELECT * FROM issues WHERE org='Acme Corp' AND project='Apollo' AND status='open'
```

**Why GLiNER handles `org_project_template` better than spaCy:** spaCy would need `org` to map to its `ORG` label, and `project` has no built-in spaCy label at all. GLiNER receives `["org", "project"]` literally and extracts the correct spans.

Custom model:

```python
# Lighter variant (~250 MB)
ext = ParameterExtractor(use_gliner=True, gliner_model="urchade/gliner_small-v2.1")

# Higher accuracy on complex sentences
ext = ParameterExtractor(use_gliner=True, gliner_model="urchade/gliner_large-v2.1")
```

> The first call downloads `urchade/gliner_medium-v2.1` (~500 MB) from HuggingFace. Subsequent runs use the local cache.

---

## Both Enabled — Cascade in Action

When both backends are enabled, each step fills in only the parameters the previous step missed.

```python
ext = ParameterExtractor(use_spacy=True, use_gliner=True)

# Regex resolves {count}, GLiNER resolves {project}
hybrid_template = QueryTemplate(
    intent="top_n_project",
    template_text="Show top {count} issues for project {project}",
    query_template="SELECT TOP {count} * FROM issues WHERE project = '{project}'",
    parameters=["count", "project"],
    parameter_patterns={"count": r"\b(\d+)\b"},
)

params = ext.extract("show top 3 issues for project Hermes", hybrid_template)
# {"count": "3", "project": "Hermes"}
```

---

## Choosing the Right Backend

| Scenario | Recommendation |
|---|---|
| Numeric or enum parameters | Regex only (`use_spacy=False, use_gliner=False`) |
| Standard entities (person, org, number) | spaCy (`use_spacy=True`) |
| Domain-specific or unpredictable param names | GLiNER (`use_gliner=True`) |
| Mixed templates in the same app | Both enabled — cascade handles it automatically |
| Edge / resource-constrained deployment | Regex + heuristics only |

Both backends fall back gracefully if the package is not installed.

---

!!! note "Full working example"
    The complete runnable notebook with latency comparisons is available at
    [`demo/05_ner_spacy_vs_gliner.ipynb`](https://github.com/ArchAI-Labs/medha/blob/main/demo/05_ner_spacy_vs_gliner.ipynb).
