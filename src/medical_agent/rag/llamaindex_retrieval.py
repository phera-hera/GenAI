"""
Medical RAG retrieval using LlamaIndex retriever.retrieve().

Provides structured node retrieval from medical research papers.
Returns NodeWithScore objects for use in LangGraph workflows.
"""

import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Retriever Builder
# ============================================================================


def build_retriever(similarity_top_k: int = 5):
    """
    Build a retriever with vector store connection.

    Args:
        similarity_top_k: Number of top chunks to retrieve

    Returns:
        Configured retriever object
    """
    # Initialize embeddings
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

    # Connect to vector store
    vector_store = PGVectorStore.from_params(
        database=settings.postgres_db,
        host=settings.postgres_host,
        password=settings.postgres_password,
        port=settings.postgres_port,
        user=settings.postgres_user,
        table_name="paper_chunks",  # LlamaIndex adds "data_" prefix
        embed_dim=settings.embedding_dimension,
        hybrid_search=True,
        text_search_config="english",
        perform_setup=False,
    )

    # Create VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Return retriever for structured node retrieval
    return index.as_retriever(similarity_top_k=similarity_top_k)


def retrieve_nodes(query: str, similarity_top_k: int = 5) -> list[NodeWithScore]:
    """
    Retrieve medical research nodes for a query.

    Args:
        query: Search query
        similarity_top_k: Number of top chunks to retrieve

    Returns:
        List of NodeWithScore objects (nodes + relevance scores)
    """
    logger.info(f"Retrieving nodes for query: {query[:100]}...")
    retriever = build_retriever(similarity_top_k=similarity_top_k)
    nodes = retriever.retrieve(query)
    logger.info(f"Retrieved {len(nodes)} nodes")
    return nodes
