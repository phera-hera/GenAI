"""
LlamaIndex Setup for Medical RAG

Configures LlamaIndex with pgvector for hybrid search combining:
- Vector similarity search (semantic)
- BM25/Full-text search (keyword-based)

Features:
- PGVectorStore integration with hybrid search
- Azure OpenAI embedding model
- Azure OpenAI LLM for response generation
- Configurable index and search settings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.core.exceptions import DatabaseException, LLMError
from app.services.azure_openai import get_llama_index_embed_model, get_llama_index_llm

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the PGVectorStore with hybrid search support."""

    # Database connection
    database: str
    host: str
    port: int
    user: str
    password: str

    # Table configuration
    table_name: str = "paper_chunks"
    schema_name: str = "public"
    embed_dim: int = 1536

    # Hybrid search settings (semantic + BM25)
    hybrid_search: bool = True  # Enable hybrid by default
    text_search_config: str = "english"  # PostgreSQL text search config

    # Performance tuning
    hnsw_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_settings(cls, hybrid_search: bool = True) -> VectorStoreConfig:
        """
        Create configuration from application settings.

        Args:
            hybrid_search: Enable hybrid search (semantic + BM25). Default True.

        Returns:
            VectorStoreConfig instance
        """
        return cls(
            database=settings.postgres_db,
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            embed_dim=settings.embedding_dimension,
            hybrid_search=hybrid_search,
        )


class MedicalRAGIndex:
    """
    LlamaIndex-based vector index for medical paper retrieval.

    Wraps PGVectorStore and provides methods for:
    - Connecting to existing vector store
    - Semantic similarity search
    - Hybrid search (vector + full-text)
    - Index statistics and health checks

    Args:
        config: VectorStoreConfig for database connection
        embed_model: Optional embedding model (uses Azure OpenAI if not provided)
        llm: Optional LLM (uses Azure OpenAI GPT-4o if not provided)
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embed_model: BaseEmbedding | None = None,
        llm: LLM | None = None,
    ):
        self.config = config or VectorStoreConfig.from_settings()
        self._embed_model = embed_model
        self._llm = llm
        self._vector_store: PGVectorStore | None = None
        self._index: VectorStoreIndex | None = None

    @property
    def embed_model(self) -> BaseEmbedding:
        """Get or create the embedding model."""
        if self._embed_model is None:
            if not settings.is_azure_openai_configured():
                raise LLMError("Azure OpenAI not configured for embeddings")
            self._embed_model = get_llama_index_embed_model()
        return self._embed_model

    @property
    def llm(self) -> LLM:
        """Get or create the LLM."""
        if self._llm is None:
            if not settings.is_azure_openai_configured():
                raise LLMError("Azure OpenAI not configured for LLM")
            self._llm = get_llama_index_llm()
        return self._llm

    @property
    def vector_store(self) -> PGVectorStore:
        """Get or create the vector store connection."""
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()
        return self._vector_store

    @property
    def index(self) -> VectorStoreIndex:
        """Get or create the vector store index."""
        if self._index is None:
            self._index = self._create_index()
        return self._index

    def _create_vector_store(self) -> PGVectorStore:
        """Create a PGVectorStore connection to existing data."""
        try:
            # Configure global LlamaIndex settings
            LlamaSettings.embed_model = self.embed_model
            LlamaSettings.llm = self.llm

            # Build connection string for PGVectorStore
            # Note: PGVectorStore uses psycopg2 (sync) internally
            vector_store = PGVectorStore.from_params(
                database=self.config.database,
                host=self.config.host,
                port=str(self.config.port),
                user=self.config.user,
                password=self.config.password,
                table_name=self.config.table_name,
                schema_name=self.config.schema_name,
                embed_dim=self.config.embed_dim,
                hybrid_search=self.config.hybrid_search,
                text_search_config=self.config.text_search_config,
                hnsw_kwargs=self.config.hnsw_kwargs,
            )

            mode = "hybrid (semantic + BM25)" if self.config.hybrid_search else "semantic only"
            logger.info(
                f"Connected to PGVectorStore: {self.config.table_name} "
                f"(dim={self.config.embed_dim}, mode={mode})"
            )
            return vector_store

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise DatabaseException(f"Failed to create vector store: {e}")

    def _create_index(self) -> VectorStoreIndex:
        """Create a VectorStoreIndex from the existing vector store."""
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Create index from existing vector store (no data insertion)
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

            logger.info("Created VectorStoreIndex from existing data")
            return index

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise DatabaseException(f"Failed to create index: {e}")

    def refresh(self) -> None:
        """Refresh the index connection (useful after data changes)."""
        self._vector_store = None
        self._index = None
        logger.info("Index connection refreshed")

    def get_retriever(
        self,
        similarity_top_k: int | None = None,
        **kwargs: Any,
    ):
        """
        Get a retriever from the index.

        Args:
            similarity_top_k: Number of top results to retrieve
            **kwargs: Additional retriever configuration

        Returns:
            VectorIndexRetriever for semantic search
        """
        top_k = similarity_top_k or settings.vector_similarity_top_k
        return self.index.as_retriever(
            similarity_top_k=top_k,
            **kwargs,
        )


# =============================================================================
# Factory Functions
# =============================================================================

_rag_index: MedicalRAGIndex | None = None


def get_medical_rag_index(
    config: VectorStoreConfig | None = None,
    force_new: bool = False,
) -> MedicalRAGIndex:
    """
    Get or create the global MedicalRAGIndex instance.

    Args:
        config: Optional custom configuration
        force_new: If True, create a new instance even if one exists

    Returns:
        MedicalRAGIndex instance
    """
    global _rag_index

    if _rag_index is None or force_new:
        _rag_index = MedicalRAGIndex(config=config)

    return _rag_index


def create_node_from_chunk(
    chunk_id: str,
    content: str,
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TextNode:
    """
    Create a LlamaIndex TextNode from chunk data.

    Useful for bridging between the existing PaperChunk model
    and LlamaIndex's node structure.

    Args:
        chunk_id: Unique identifier for the chunk
        content: Text content of the chunk
        embedding: Optional pre-computed embedding
        metadata: Optional metadata dict

    Returns:
        TextNode ready for LlamaIndex operations
    """
    node = TextNode(
        id_=chunk_id,
        text=content,
        embedding=embedding,
        metadata=metadata or {},
    )
    return node


def create_node_with_score(
    chunk_id: str,
    content: str,
    score: float,
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> NodeWithScore:
    """
    Create a LlamaIndex NodeWithScore from chunk data and similarity score.

    Args:
        chunk_id: Unique identifier for the chunk
        content: Text content of the chunk
        score: Similarity score
        embedding: Optional pre-computed embedding
        metadata: Optional metadata dict

    Returns:
        NodeWithScore for retrieval results
    """
    node = create_node_from_chunk(
        chunk_id=chunk_id,
        content=content,
        embedding=embedding,
        metadata=metadata,
    )
    return NodeWithScore(node=node, score=score)

