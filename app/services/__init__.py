"""
Business logic services.

This module provides client wrappers for all external services:
- GCP Cloud Storage for PDF storage
- Azure Document Intelligence for PDF parsing
- Azure OpenAI for LLM and embeddings
- Langfuse for observability
"""

from app.services.azure_document import (
    AzureDocumentClient,
    get_document_client,
)
from app.services.azure_openai import (
    AzureOpenAIClient,
    get_llama_index_embed_model,
    get_llama_index_llm,
    get_openai_client,
)
from app.services.gcp_storage import (
    GCPStorageClient,
    get_storage_client,
)
from app.services.langfuse_client import (
    LangfuseClient,
    get_langfuse_client,
    trace_operation,
)

__all__ = [
    # GCP Storage
    "GCPStorageClient",
    "get_storage_client",
    # Azure Document Intelligence
    "AzureDocumentClient",
    "get_document_client",
    # Azure OpenAI
    "AzureOpenAIClient",
    "get_openai_client",
    "get_llama_index_llm",
    "get_llama_index_embed_model",
    # Langfuse
    "LangfuseClient",
    "get_langfuse_client",
    "trace_operation",
]
