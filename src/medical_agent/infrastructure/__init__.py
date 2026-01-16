"""
Business logic services.

This module provides client wrappers for all external services:
- GCP Cloud Storage for PDF storage
- LlamaParser for PDF parsing
- Azure OpenAI for LLM and embeddings
- Langfuse for observability
"""

from medical_agent.infrastructure.llama_parser import (
    LlamaParserClient,
    get_llama_parser_client,
)
from medical_agent.infrastructure.azure_openai import (
    AzureOpenAIClient,
    get_llama_index_embed_model,
    get_llama_index_llm,
    get_openai_client,
)
from medical_agent.infrastructure.gcp_storage import (
    GCPStorageClient,
    get_storage_client,
)
from medical_agent.infrastructure.langfuse_client import (
    LangfuseClient,
    get_langfuse_client,
    trace_operation,
)

__all__ = [
    # GCP Storage
    "GCPStorageClient",
    "get_storage_client",
    # LlamaParser
    "LlamaParserClient",
    "get_llama_parser_client",
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
