"""Azure OpenAI embedding generation for medical paper chunks."""

from .azure_embedder import (
    AsyncAzureEmbedder,
    AzureEmbedder,
    EmbeddedChunk,
    EmbeddingResult,
    get_async_embedder,
    get_embedder,
)

__all__ = [
    "AzureEmbedder",
    "AsyncAzureEmbedder",
    "EmbeddedChunk",
    "EmbeddingResult",
    "get_embedder",
    "get_async_embedder",
]
