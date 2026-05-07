# Exceptions

All Medha exceptions inherit from `MedhaError`, making it easy to catch any library error with a single handler.

```python
from medha.exceptions import MedhaError

try:
    async with Medha(...) as cache:
        await cache.store(...)
except MedhaError as exc:
    logger.error("Medha error: %s", exc)
```

---

## Exception Hierarchy

| Exception | Raised When |
|---|---|
| `MedhaError` | Base class for all Medha exceptions |
| `ConfigurationError` | Invalid `Settings` values (e.g. threshold ordering violation) |
| `EmbeddingError` | Embedder fails to produce a vector (API error, model load failure) |
| `StorageError` | Vector backend read/write operation fails |
| `StorageInitializationError` | Backend cannot connect or create the collection at startup |
| `TemplateError` | A `QueryTemplate` is malformed or cannot be registered |
| `ParameterExtractionError` | Named entity extraction fails during template matching |

---

## MedhaError

::: medha.exceptions.MedhaError

---

## ConfigurationError

::: medha.exceptions.ConfigurationError

---

## EmbeddingError

::: medha.exceptions.EmbeddingError

---

## StorageError

::: medha.exceptions.StorageError

---

## StorageInitializationError

::: medha.exceptions.StorageInitializationError

---

## TemplateError

::: medha.exceptions.TemplateError

---

## ParameterExtractionError

::: medha.exceptions.ParameterExtractionError
