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
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.config import settings
from medical_agent.core.exceptions import DatabaseException
from medical_agent.infrastructure.database.models import ChunkType, Paper, PaperChunk
from medical_agent.ingestion.storage.metadata_filters import build_metadata_filters, format_metadata_filters
from medical_agent.ingestion.storage.types import SearchQuery, SearchResult

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
        session: AsyncSession,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            session: Database session
            query: Search configuration

        Returns:
            List of SearchResult sorted by similarity (most similar first)
        """
        try:
            # Build the distance expression based on metric
            if query.distance_metric == "cosine":
                # Cosine distance: 1 - cosine_similarity
                # We want similarity, so we use 1 - distance
                distance_expr = PaperChunk.embedding.cosine_distance(query.embedding)
                score_expr = 1 - distance_expr
            elif query.distance_metric == "l2":
                # L2 distance: smaller is more similar
                # Convert to similarity: 1 / (1 + distance)
                distance_expr = PaperChunk.embedding.l2_distance(query.embedding)
                score_expr = 1 / (1 + distance_expr)
            else:  # inner_product
                # Inner product: larger is more similar (for normalized vectors)
                score_expr = PaperChunk.embedding.max_inner_product(query.embedding)

            # Build base query
            stmt = (
                select(
                    PaperChunk.id,
                    PaperChunk.paper_id,
                    PaperChunk.content,
                    PaperChunk.chunk_type,
                    PaperChunk.section_title,
                    PaperChunk.page_number,
                    PaperChunk.chunk_metadata,
                    score_expr.label("score"),
                )
                .where(PaperChunk.embedding.isnot(None))
            )

            # Apply filters
            if query.paper_ids:
                stmt = stmt.where(PaperChunk.paper_id.in_(query.paper_ids))

            if query.chunk_types:
                type_values = [ct.value for ct in query.chunk_types]
                stmt = stmt.where(PaperChunk.chunk_type.in_(type_values))

            # Apply medical metadata filters
            metadata_filters = build_metadata_filters(
                filter_ethnicities=query.filter_ethnicities,
                filter_diagnoses=query.filter_diagnoses,
                filter_symptoms=query.filter_symptoms,
                filter_menstrual_status=query.filter_menstrual_status,
                filter_birth_control=query.filter_birth_control,
                filter_hormone_therapy=query.filter_hormone_therapy,
                filter_fertility_treatments=query.filter_fertility_treatments,
            )

            if metadata_filters:
                stmt = stmt.where(and_(*metadata_filters))
                logger.debug(
                    f"Applied metadata filters: {format_metadata_filters(
                        filter_ethnicities=query.filter_ethnicities,
                        filter_diagnoses=query.filter_diagnoses,
                        filter_symptoms=query.filter_symptoms,
                        filter_menstrual_status=query.filter_menstrual_status,
                        filter_birth_control=query.filter_birth_control,
                        filter_hormone_therapy=query.filter_hormone_therapy,
                        filter_fertility_treatments=query.filter_fertility_treatments,
                    )}"
                )

            # Order by similarity and limit
            if query.distance_metric in ("cosine", "l2"):
                # For cosine and L2, we order by distance (ascending)
                stmt = stmt.order_by(distance_expr).limit(query.top_k)
            else:
                # For inner product, order by score descending
                stmt = stmt.order_by(score_expr.desc()).limit(query.top_k)

            # Execute
            result = await session.execute(stmt)
            rows = result.fetchall()

            # Build results
            search_results = []
            for row in rows:
                score = float(row.score) if row.score is not None else 0.0

                # Apply minimum score filter
                if query.min_score is not None and score < query.min_score:
                    continue

                search_result = SearchResult(
                    chunk_id=row.id,
                    paper_id=row.paper_id,
                    content=row.content,
                    chunk_type=row.chunk_type,
                    section_title=row.section_title,
                    page_number=row.page_number,
                    chunk_metadata=row.chunk_metadata or {},
                    score=score,
                )
                search_results.append(search_result)

            # Fetch paper metadata if requested
            if query.include_paper_metadata and search_results:
                paper_ids = list({r.paper_id for r in search_results})
                paper_stmt = select(
                    Paper.id,
                    Paper.title,
                    Paper.authors,
                    Paper.doi,
                ).where(Paper.id.in_(paper_ids))

                paper_result = await session.execute(paper_stmt)
                paper_map = {r.id: r for r in paper_result.fetchall()}

                for sr in search_results:
                    if sr.paper_id in paper_map:
                        paper = paper_map[sr.paper_id]
                        sr.paper_title = paper.title
                        sr.paper_authors = paper.authors
                        sr.paper_doi = paper.doi

            logger.debug(f"Similarity search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise DatabaseException(f"Similarity search failed: {e}")

    async def bm25_search(
        self,
        session: AsyncSession,
        query_text: str,
        top_k: int = 10,
        paper_ids: list[uuid.UUID] | None = None,
        chunk_types: list[ChunkType] | None = None,
        include_paper_metadata: bool = True,
        # Medical metadata filters
        filter_ethnicities: list[str] | None = None,
        filter_diagnoses: list[str] | None = None,
        filter_symptoms: list[str] | None = None,
        filter_menstrual_status: list[str] | None = None,
        filter_birth_control: list[str] | None = None,
        filter_hormone_therapy: list[str] | None = None,
        filter_fertility_treatments: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Perform BM25 full-text search using PostgreSQL's tsvector/tsquery.

        Args:
            session: Database session
            query_text: Text query for full-text search
            top_k: Number of results to return
            paper_ids: Optional filter by paper IDs
            chunk_types: Optional filter by chunk types
            include_paper_metadata: Whether to fetch paper metadata
            filter_ethnicities: Filter by ethnicity values
            filter_diagnoses: Filter by diagnosis values
            filter_symptoms: Filter by symptom values
            filter_menstrual_status: Filter by menstrual status values
            filter_birth_control: Filter by birth control types
            filter_hormone_therapy: Filter by hormone therapy types
            filter_fertility_treatments: Filter by fertility treatment types

        Returns:
            List of SearchResult sorted by text relevance
        """
        try:
            # Build the full-text search query
            # PostgreSQL ts_rank gives relevance score
            search_query = func.plainto_tsquery("english", query_text)
            tsvector_col = func.to_tsvector("english", PaperChunk.content)
            rank_expr = func.ts_rank(tsvector_col, search_query)

            # Build base query with full-text match
            stmt = (
                select(
                    PaperChunk.id,
                    PaperChunk.paper_id,
                    PaperChunk.content,
                    PaperChunk.chunk_type,
                    PaperChunk.section_title,
                    PaperChunk.page_number,
                    PaperChunk.chunk_metadata,
                    rank_expr.label("score"),
                )
                .where(tsvector_col.op("@@")(search_query))
            )

            # Apply filters
            if paper_ids:
                stmt = stmt.where(PaperChunk.paper_id.in_(paper_ids))

            if chunk_types:
                type_values = [ct.value for ct in chunk_types]
                stmt = stmt.where(PaperChunk.chunk_type.in_(type_values))

            # Apply medical metadata filters
            metadata_filters = build_metadata_filters(
                filter_ethnicities=filter_ethnicities,
                filter_diagnoses=filter_diagnoses,
                filter_symptoms=filter_symptoms,
                filter_menstrual_status=filter_menstrual_status,
                filter_birth_control=filter_birth_control,
                filter_hormone_therapy=filter_hormone_therapy,
                filter_fertility_treatments=filter_fertility_treatments,
            )

            if metadata_filters:
                stmt = stmt.where(and_(*metadata_filters))
                logger.debug(
                    f"BM25 search with metadata filters: {format_metadata_filters(
                        filter_ethnicities=filter_ethnicities,
                        filter_diagnoses=filter_diagnoses,
                        filter_symptoms=filter_symptoms,
                        filter_menstrual_status=filter_menstrual_status,
                        filter_birth_control=filter_birth_control,
                        filter_hormone_therapy=filter_hormone_therapy,
                        filter_fertility_treatments=filter_fertility_treatments,
                    )}"
                )

            # Order by relevance and limit
            stmt = stmt.order_by(rank_expr.desc()).limit(top_k)

            # Execute
            result = await session.execute(stmt)
            rows = result.fetchall()

            # Build results
            search_results = []
            for row in rows:
                score = float(row.score) if row.score is not None else 0.0
                search_result = SearchResult(
                    chunk_id=row.id,
                    paper_id=row.paper_id,
                    content=row.content,
                    chunk_type=row.chunk_type,
                    section_title=row.section_title,
                    page_number=row.page_number,
                    chunk_metadata=row.chunk_metadata or {},
                    score=score,
                )
                search_results.append(search_result)

            # Fetch paper metadata if requested
            if include_paper_metadata and search_results:
                paper_ids_list = list({r.paper_id for r in search_results})
                paper_stmt = select(
                    Paper.id,
                    Paper.title,
                    Paper.authors,
                    Paper.doi,
                ).where(Paper.id.in_(paper_ids_list))

                paper_result = await session.execute(paper_stmt)
                paper_map = {r.id: r for r in paper_result.fetchall()}

                for sr in search_results:
                    if sr.paper_id in paper_map:
                        paper = paper_map[sr.paper_id]
                        sr.paper_title = paper.title
                        sr.paper_authors = paper.authors
                        sr.paper_doi = paper.doi

            logger.debug(f"BM25 search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise DatabaseException(f"BM25 search failed: {e}")

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
