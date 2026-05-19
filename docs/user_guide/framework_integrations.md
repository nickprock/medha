# Framework Integrations

Medha can be plugged into **LangChain**, **LlamaIndex**, and **Haystack** using each framework's native extension point. The cache is transparent to the rest of the pipeline: a hit short-circuits before the LLM is called; a miss falls through normally and the result is stored for next time.

| Framework | Integration point | Medha role |
|---|---|---|
| **LangChain** | `langchain_core.caches.BaseCache` | Intercepts LLM calls via `set_llm_cache()` |
| **LlamaIndex** | Workflow `@step` | Pre-filter before the query engine |
| **Haystack** | `@component` in a `Pipeline` | First stage, short-circuits on cache hit |

All examples use `InMemoryBackend` — no Qdrant, no PostgreSQL, no API keys required.

**Requirements:**

```bash
pip install "medha-archai[fastembed]"
pip install langchain-core langchain-community
pip install llama-index-core
pip install haystack-ai
```

---

## Shared Setup

A single `Medha` instance and a mock SQL generator used by all three integrations.

```python
import asyncio
from medha import Medha, Settings, SearchStrategy
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

embedder = FastEmbedAdapter(model_name="BAAI/bge-small-en-v1.5")
settings = Settings(backend_type="memory", score_threshold_semantic=0.82)
medha = Medha("framework_demo", embedder=embedder, settings=settings)
await medha.start()

# Mock LLM — simulates a Text-to-SQL model (no API key needed)
_SQL_MAP = {
    "users":    "SELECT COUNT(*) FROM users",
    "revenue":  "SELECT SUM(amount) FROM invoices WHERE YEAR(created_at) = YEAR(NOW())",
    "products": "SELECT * FROM products ORDER BY price DESC LIMIT 10",
    "orders":   "SELECT COUNT(*) FROM orders WHERE DATE(created_at) = CURDATE()",
    "salary":   "SELECT AVG(salary) FROM employees",
}

def mock_llm(question: str) -> str:
    q = question.lower()
    for keyword, sql in _SQL_MAP.items():
        if keyword in q:
            return sql
    return f"SELECT * FROM unknown  -- question: {question}"

# Warm the cache with known pairs
seed_pairs = [
    ("How many users are registered?",          "SELECT COUNT(*) FROM users"),
    ("What is the total revenue this year?",    "SELECT SUM(amount) FROM invoices WHERE YEAR(created_at) = YEAR(NOW())"),
    ("Show the top 10 most expensive products", "SELECT * FROM products ORDER BY price DESC LIMIT 10"),
    ("Count orders placed today",               "SELECT COUNT(*) FROM orders WHERE DATE(created_at) = CURDATE()"),
    ("What is the average employee salary?",    "SELECT AVG(salary) FROM employees"),
]
for q, sql in seed_pairs:
    await medha.store(q, sql)
```

---

## LangChain

LangChain exposes `BaseCache` as the standard cache interface. Any object implementing `lookup()` / `update()` / `clear()` can be registered globally with `set_llm_cache()` — every subsequent LLM call checks the cache first.

```
chain.invoke(question)
    └─ LLM.__call__(prompt)
           └─ cache.lookup(prompt, llm_string)   ← Medha.search()
                  hit  → return cached Generation
                  miss → call LLM → cache.update() → return
```

```python
import concurrent.futures
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation
from langchain_core.globals import set_llm_cache
from typing import Optional


def _run_async(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class MedhaLangChainCache(BaseCache):
    def __init__(self, medha_instance: Medha) -> None:
        self._medha = medha_instance

    def lookup(self, prompt: str, llm_string: str) -> Optional[list[Generation]]:
        hit = _run_async(self._medha.search(prompt))
        if hit.strategy == SearchStrategy.NO_MATCH:
            return None
        return [Generation(text=hit.generated_query)]

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        sql = return_val[0].text if return_val else ""
        _run_async(self._medha.store(prompt, sql))

    def clear(self, **kwargs) -> None:
        pass


set_llm_cache(MedhaLangChainCache(medha))
```

---

## LlamaIndex

LlamaIndex 0.10+ Workflows use typed `Event` objects and `@step` decorated async methods. The cache check is a first step that short-circuits to `StopEvent` on a hit, or emits `CacheMissEvent` to trigger the LLM step.

```
StartEvent(question)
    └─ check_cache ──hit──→ StopEvent(sql)
                   ──miss─→ CacheMissEvent(question)
                                 └─ call_llm → store → StopEvent(sql)
```

```python
from llama_index.core.workflow import Workflow, step, Event, StartEvent, StopEvent, Context
from typing import Union


class CacheMissEvent(Event):
    question: str


class MedhaWorkflow(Workflow):
    def __init__(self, medha_instance: Medha, **kwargs) -> None:
        super().__init__(**kwargs)
        self._medha = medha_instance
        self.llm_call_count: int = 0

    @step
    async def check_cache(self, ctx: Context, ev: StartEvent) -> Union[CacheMissEvent, StopEvent]:
        hit = await self._medha.search(ev.question)
        if hit.strategy != SearchStrategy.NO_MATCH:
            return StopEvent(result=hit.generated_query)
        return CacheMissEvent(question=ev.question)

    @step
    async def call_llm(self, ctx: Context, ev: CacheMissEvent) -> StopEvent:
        self.llm_call_count += 1
        sql = mock_llm(ev.question)
        await self._medha.store(ev.question, sql)
        return StopEvent(result=sql)


workflow = MedhaWorkflow(medha, timeout=30, verbose=False)
result = await workflow.run(question="Top products ranked by cost")
print(result)  # SELECT * FROM products ORDER BY price DESC LIMIT 10
```

---

## Haystack

Haystack 2.x uses a declarative `@component` model. The `MedhaCacheComponent` returns the cached SQL if found, or forwards the question to the LLM component.

```
question → MedhaCacheComponent
                ├─ hit  → HaystackMockLLMComponent (sql passthrough) → output
                └─ miss → HaystackMockLLMComponent (calls LLM + stores) → output
```

```python
from haystack import component, Pipeline
from typing import Optional


@component
class HaystackMedhaCacheComponent:
    def __init__(self, medha_instance: Medha) -> None:
        self._medha = medha_instance

    @component.output_types(sql=Optional[str], question=str)
    def run(self, question: str) -> dict:
        hit = _run_async(self._medha.search(question))
        if hit.strategy != SearchStrategy.NO_MATCH:
            return {"sql": hit.generated_query, "question": question}
        return {"sql": None, "question": question}


@component
class HaystackMockLLMComponent:
    def __init__(self, medha_instance: Medha) -> None:
        self._medha = medha_instance

    @component.output_types(sql=str)
    def run(self, question: str, sql: Optional[str] = None) -> dict:
        if sql:
            return {"sql": sql}
        result = mock_llm(question)
        _run_async(self._medha.store(question, result))
        return {"sql": result}


pipeline = Pipeline()
pipeline.add_component("cache", HaystackMedhaCacheComponent(medha))
pipeline.add_component("llm",   HaystackMockLLMComponent(medha))
pipeline.connect("cache.question", "llm.question")
pipeline.connect("cache.sql",      "llm.sql")

result = pipeline.run({"cache": {"question": "Top products ranked by cost"}})
print(result["llm"]["sql"])  # SELECT * FROM products ORDER BY price DESC LIMIT 10
```

---

## Production Notes

| Topic | Recommendation |
|---|---|
| **Backend** | Replace `InMemoryBackend` with a persistent backend for durability across restarts |
| **Async in sync frameworks** | LangChain and Haystack are sync-first; wrap async Medha calls with a `ThreadPoolExecutor` as shown above |
| **LlamaIndex async** | LlamaIndex 0.10+ supports `arun()` — prefer async steps in production |
| **Shared instance** | One `Medha` instance across all frameworks is fine — all backends are thread-safe |
| **TTL** | Set `default_ttl_seconds` in `Settings` to auto-expire stale queries after schema changes |
| **Embedder** | `FastEmbedAdapter` runs locally (no API cost). For multilingual queries, switch to `CohereAdapter` or `GeminiAdapter` |

### Switching to a persistent backend

```python
settings = Settings(
    backend_type="qdrant",   # or "pgvector", "elasticsearch", "lancedb", etc.
    qdrant_mode="docker",
    qdrant_host="localhost",
)
medha = Medha("production_cache", embedder=embedder, settings=settings)
await medha.start()
```

See [Backends](backends.md) for the full list of available backends and their configuration options.

!!! note "Full working example"
    The complete runnable notebook is available at
    [`demo/13_framework_integrations.ipynb`](https://github.com/ArchAI-Labs/medha/blob/main/demo/13_framework_integrations.ipynb).
