"""
LlamaIndex PGVector Store Adapter

Adapter between LlamaIndex PGVectorStore and existing paper_chunks schema.
Provides compatibility layer to use LlamaIndex's vector store while maintaining
compatibility with existing SearchResult format and retrieval system.
"""

import logging
import uuid
from typing import Any

from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.config import settings
from medical_agent.ingestion.storage.vector_store import SearchResult

logger = logging.getLogger(__name__)


class MedicalPGVectorStore:
    """
    Wrapper around LlamaIndex PGVectorStore that:
    1. Uses existing paper_chunks table
    2. Preserves chunk_metadata JSONB structure
    3. Provides SearchResult compatibility for retrieval
    4. Maintains async operations
    """

    def __init__(self, session: AsyncSession | None = None):
        """
        Initialize the LlamaIndex PGVector store adapter.

        Args:
            session: Optional AsyncSession for database operations
        """
        self._external_session = session
        self._vector_store: PGVectorStore | None = None

    def _get_vector_store(self) -> PGVectorStore:
        """
        Get or create the LlamaIndex PGVectorStore instance.

        Returns:
            Configured PGVectorStore instance
        """
        if self._vector_store is None:
            # Get connection string (sync version for psycopg2)
            # LlamaIndex PGVectorStore uses psycopg2, not asyncpg
            connection_string = settings.sync_database_connection_string

            self._vector_store = PGVectorStore.from_params(
                database=settings.postgres_db,
                host=settings.postgres_host,
                password=settings.postgres_password,
                port=settings.postgres_port,
                user=settings.postgres_user,
                table_name="paper_chunks",
                embed_dim=settings.embedding_dimension,
                # Map field names to existing schema
                text_key="content",
                embedding_key="embedding",
                metadata_key="chunk_metadata",
            )
            logger.info("Initialized LlamaIndex PGVectorStore with existing schema")

        return self._vector_store

    async def add_nodes(
        self,
        nodes: list[BaseNode],
        paper_id: uuid.UUID,
    ) -> list[str]:
        """
        Add nodes to the vector store with paper_id.

        Args:
            nodes: List of LlamaIndex nodes to store
            paper_id: UUID of the parent paper

        Returns:
            List of node IDs that were stored
        """
        if not nodes:
            return []

        try:
            # Add paper_id to all nodes' metadata
            for node in nodes:
                if node.metadata is None:
                    node.metadata = {}
                node.metadata["paper_id"] = str(paper_id)

                # Ensure node has a UUID
                if not node.id_ or not self._is_valid_uuid(node.id_):
                    node.id_ = str(uuid.uuid4())

            # Store using LlamaIndex
            vector_store = self._get_vector_store()
            node_ids = vector_store.add(nodes)

            logger.info(f"Stored {len(node_ids)} nodes for paper {paper_id}")
            return node_ids

        except Exception as e:
            logger.error(f"Failed to store nodes: {e}", exc_info=True)
            raise

    def _is_valid_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False

    async def delete_paper_chunks(
        self,
        paper_id: uuid.UUID,
    ) -> int:
        """
        Delete all chunks for a paper.

        Args:
            paper_id: UUID of the paper

        Returns:
            Number of deleted chunks
        """
        try:
            vector_store = self._get_vector_store()

            # LlamaIndex delete by filter
            # Note: This uses the metadata filter to find chunks with matching paper_id
            filters = {"paper_id": str(paper_id)}
            vector_store.delete(filters=filters)

            logger.info(f"Deleted chunks for paper {paper_id}")
            return 0  # LlamaIndex doesn't return count

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}", exc_info=True)
            raise

    def to_search_result(
        self,
        node_with_score,
        paper_metadata: dict[str, Any] | None = None,
    ) -> SearchResult:
        """
        Convert LlamaIndex VectorStoreQueryResult node to SearchResult.

        Args:
            node_with_score: NodeWithScore from LlamaIndex query result
            paper_metadata: Optional paper metadata (title, authors, doi)

        Returns:
            SearchResult compatible with existing retrieval system
        """
        node = node_with_score.node
        metadata = node.metadata or {}

        # Extract required fields
        chunk_id = uuid.UUID(node.id_) if self._is_valid_uuid(node.id_) else uuid.uuid4()
        paper_id_str = metadata.get("paper_id", "00000000-0000-0000-0000-000000000000")
        paper_id = uuid.UUID(paper_id_str) if self._is_valid_uuid(paper_id_str) else uuid.uuid4()

        # Create SearchResult
        search_result = SearchResult(
            chunk_id=chunk_id,
            paper_id=paper_id,
            content=node.get_content(),
            chunk_type=metadata.get("chunk_type", "other"),
            section_title=metadata.get("section_title"),
            page_number=metadata.get("page_number"),
            chunk_metadata=metadata,
            score=node_with_score.score or 0.0,
        )

        # Add paper metadata if provided
        if paper_metadata:
            search_result.paper_title = paper_metadata.get("title")
            search_result.paper_authors = paper_metadata.get("authors")
            search_result.paper_doi = paper_metadata.get("doi")

        return search_result

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        try:
            from llama_index.core.vector_stores import VectorStoreQuery

            vector_store = self._get_vector_store()

            # Build query
            query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=filters,
            )

            # Execute search
            result = vector_store.query(query)

            # Convert to SearchResult objects
            search_results = []
            for node_with_score in result.nodes:
                search_result = self.to_search_result(node_with_score)
                search_results.append(search_result)

            logger.debug(f"Similarity search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise

    async def get_chunk_count(
        self,
        paper_id: uuid.UUID | None = None,
    ) -> int:
        """
        Get total chunk count, optionally filtered by paper.

        Args:
            paper_id: Optional paper ID to filter by

        Returns:
            Number of chunks
        """
        # Note: LlamaIndex PGVectorStore doesn't provide a direct count method
        # This would require a custom query
        logger.warning("get_chunk_count not fully implemented for LlamaIndex store")
        return 0


def get_llamaindex_vector_store(session: AsyncSession | None = None) -> MedicalPGVectorStore:
    """
    Factory function to create a MedicalPGVectorStore instance.

    Args:
        session: Optional AsyncSession

    Returns:
        Configured MedicalPGVectorStore
    """
    return MedicalPGVectorStore(session=session)
