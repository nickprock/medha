"""Embedder adapter implementations (FastEmbed, OpenAI, Cohere, Gemini).

Lazy imports to avoid pulling in optional dependencies at package level.
"""

from medha.interfaces.embedder import BaseEmbedder


def get_fastembed_adapter() -> type[BaseEmbedder]:
    """Import and return the FastEmbedAdapter class."""
    from medha.embeddings.fastembed_adapter import FastEmbedAdapter

    return FastEmbedAdapter


def get_openai_adapter() -> type[BaseEmbedder]:
    """Import and return the OpenAIAdapter class."""
    from medha.embeddings.openai_adapter import OpenAIAdapter

    return OpenAIAdapter


def get_cohere_adapter() -> type[BaseEmbedder]:
    """Import and return the CohereAdapter class."""
    from medha.embeddings.cohere_adapter import CohereAdapter

    return CohereAdapter


def get_gemini_adapter() -> type[BaseEmbedder]:
    """Import and return the GeminiAdapter class."""
    from medha.embeddings.gemini_adapter import GeminiAdapter

    return GeminiAdapter


__all__ = ["get_fastembed_adapter", "get_openai_adapter", "get_cohere_adapter", "get_gemini_adapter"]
