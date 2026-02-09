"""Embedder adapter implementations (FastEmbed, OpenAI).

Lazy imports to avoid pulling in optional dependencies at package level.
"""


def get_fastembed_adapter():
    """Import and return the FastEmbedAdapter class."""
    from medha.embeddings.fastembed_adapter import FastEmbedAdapter

    return FastEmbedAdapter


def get_openai_adapter():
    """Import and return the OpenAIAdapter class."""
    from medha.embeddings.openai_adapter import OpenAIAdapter

    return OpenAIAdapter


__all__ = ["get_fastembed_adapter", "get_openai_adapter"]
