# Embedders

Medha is embedder-agnostic. You plug in any implementation of `BaseEmbedder` at construction time. Four adapters are provided out of the box.

---

## Comparison

| Embedder | Dimensions | API Key | Cost | Best For |
|---|---|---|---|---|
| `FastEmbedAdapter` | 384–768 | No | Free | Dev, on-prem, air-gapped |
| `OpenAIAdapter` | 1536–3072 | Yes | $ | General purpose |
| `CohereAdapter` | 1024 | Yes | $$ | Multilingual, enterprise |
| `GeminiAdapter` | 768 | Yes | $ | Google ecosystem |

!!! tip

    You can switch embedders at any time, but existing cache entries will no longer match because their stored embeddings were produced by the old model. See the [FAQ](../faq.md) for a migration strategy.

---

## `FastEmbedAdapter` — Local ONNX

!!! info "Install"

    ```bash
    pip install "medha-archai[fastembed]"
    ```

FastEmbed runs ONNX models in-process with no API key, no network calls (after first model download), and no per-token cost. The default model is `BAAI/bge-small-en-v1.5` (384 dimensions).

```python
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

# Default model
embedder = FastEmbedAdapter()

# Larger model for higher accuracy
embedder = FastEmbedAdapter(model_name="BAAI/bge-large-en-v1.5")
```

**Async usage:**

```python
import asyncio
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    embedder = FastEmbedAdapter()
    vector = await embedder.embed("How many active users do we have?")
    print(len(vector))  # 384

asyncio.run(main())
```

---

## `OpenAIAdapter` — OpenAI

!!! info "Install"

    ```bash
    pip install "medha-archai[openai]"
    ```

Uses the OpenAI Embeddings API. The default model is `text-embedding-3-small` (1536 dimensions); `text-embedding-3-large` gives 3072 dimensions.

```python
import os
from medha.embeddings.openai_adapter import OpenAIAdapter

embedder = OpenAIAdapter(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
```

**Async usage:**

```python
import asyncio, os
from medha.embeddings.openai_adapter import OpenAIAdapter

async def main():
    embedder = OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"])
    vector = await embedder.embed("How many active users do we have?")
    print(len(vector))  # 1536

asyncio.run(main())
```

---

## `CohereAdapter` — Cohere

!!! info "Install"

    ```bash
    pip install "medha-archai[cohere]"
    ```

Uses Cohere's Embed v3 API. Produces 1024-dimensional embeddings with strong multilingual and domain-specific performance.

```python
import os
from medha.embeddings.cohere_adapter import CohereAdapter

embedder = CohereAdapter(
    api_key=os.environ["COHERE_API_KEY"],
    model="embed-english-v3.0",      # or embed-multilingual-v3.0
    input_type="search_query",       # or "search_document" for stored entries
)
```

**Async usage:**

```python
import asyncio, os
from medha.embeddings.cohere_adapter import CohereAdapter

async def main():
    embedder = CohereAdapter(api_key=os.environ["COHERE_API_KEY"])
    vector = await embedder.embed("How many active users do we have?")
    print(len(vector))  # 1024

asyncio.run(main())
```

---

## `GeminiAdapter` — Google Gemini

!!! info "Install"

    ```bash
    pip install "medha-archai[gemini]"
    ```

Uses the Google Generative AI Embeddings API. Produces 768-dimensional embeddings and integrates naturally with Google Cloud infrastructure.

```python
import os
from medha.embeddings.gemini_adapter import GeminiAdapter

embedder = GeminiAdapter(
    api_key=os.environ["GOOGLE_API_KEY"],
    model="models/embedding-001",
)
```

**Async usage:**

```python
import asyncio, os
from medha.embeddings.gemini_adapter import GeminiAdapter

async def main():
    embedder = GeminiAdapter(api_key=os.environ["GOOGLE_API_KEY"])
    vector = await embedder.embed("How many active users do we have?")
    print(len(vector))  # 768

asyncio.run(main())
```

---

## Custom Embedder

Implement `BaseEmbedder` to plug in any embedding model:

```python
from medha.interfaces.embedder import BaseEmbedder


class MyCustomEmbedder(BaseEmbedder):
    """Wraps any embedding model."""

    def __init__(self, model):
        self._model = model

    async def embed(self, text: str) -> list[float]:
        # Must return a list of floats (the embedding vector)
        return await self._model.encode_async(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Optional: override for batched efficiency
        return await self._model.encode_batch_async(texts)
```

Then pass it to `Medha` exactly like any built-in adapter:

```python
from medha import Medha, Settings

embedder = MyCustomEmbedder(my_model)
settings = Settings(backend_type="memory")

async with Medha("demo", embedder=embedder, settings=settings) as cache:
    ...
```

See [Interfaces (ABCs)](../api/interfaces.md) for the full `BaseEmbedder` contract.
