"""
pgvector Storage for Medical Paper Chunks

Provides storage and retrieval operations for paper chunks with
vector embeddings in PostgreSQL using pgvector.

Features:
- Store chunks with embeddings
- Vector similarity search (cosine, L2, inner product)
- Filtered search by paper, chunk type, metadata
- Batch operations for efficient ingestion
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import and_, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.config import settings
from medical_agent.core.exceptions import DatabaseException
from medical_agent.infrastructure.database.models import ChunkType, Paper, PaperChunk
from medical_agent.infrastructure.database.session import get_session_context

if TYPE_CHECKING:
    from ingestion.embedders.azure_embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


# Vector distance metrics supported by pgvector
DistanceMetric = Literal["cosine", "l2", "inner_product"]


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    content: str
    chunk_type: str
    section_title: str | None
    page_number: int | None
    chunk_metadata: dict[str, Any]
    score: float  # Similarity score (higher is more similar for cosine)
    paper_title: str | None = None
    paper_authors: str | None = None
    paper_doi: str | None = None

    @property
    def citation(self) -> str:
        """Generate a basic citation string."""
        parts = []
        if self.paper_authors:
            parts.append(self.paper_authors.split(",")[0].strip() + " et al.")
        if self.paper_title:
            parts.append(f'"{self.paper_title}"')
        if self.paper_doi:
            parts.append(f"DOI: {self.paper_doi}")
        return " - ".join(parts) if parts else f"Paper ID: {self.paper_id}"


@dataclass
class SearchQuery:
    """Configuration for a vector similarity search."""

    embedding: list[float]
    top_k: int = 10
    distance_metric: DistanceMetric = "cosine"
    paper_ids: list[uuid.UUID] | None = None
    chunk_types: list[ChunkType] | None = None
    min_score: float | None = None
    include_paper_metadata: bool = True


@dataclass
class StorageResult:
    """Result of a storage operation."""

    stored_count: int = 0
    failed_count: int = 0
    chunk_ids: list[uuid.UUID] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_successful(self) -> bool:
        return self.failed_count == 0


class VectorStore:
    """
    pgvector-based storage for medical paper chunks.

    Provides async operations for storing and retrieving paper chunks
    with vector embeddings.

    Args:
        session: Optional AsyncSession. If not provided, creates sessions
            per operation using the context manager.
    """

    def __init__(self, session: AsyncSession | None = None):
        self._external_session = session

    async def _get_session(self) -> AsyncSession:
        """Get or create a database session."""
        if self._external_session:
            return self._external_session
        raise DatabaseException("No session provided. Use context manager methods.")

    # =========================================================================
    # Storage Operations
    # =========================================================================

    async def store_chunk(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        embedded_chunk: EmbeddedChunk,
    ) -> uuid.UUID:
        """
        Store a single embedded chunk.

        Args:
            session: Database session
            paper_id: ID of the parent paper
            embedded_chunk: Chunk with embedding

        Returns:
            ID of the stored chunk

        Raises:
            DatabaseException: If storage fails
        """
        try:
            chunk = embedded_chunk.chunk

            paper_chunk = PaperChunk(
                paper_id=paper_id,
                content=chunk.content,
                chunk_type=chunk.chunk_type.value,
                chunk_index=chunk.chunk_index,
                embedding=embedded_chunk.embedding,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                chunk_metadata=chunk.chunk_metadata,
            )

            session.add(paper_chunk)
            await session.flush()

            logger.debug(f"Stored chunk {paper_chunk.id} for paper {paper_id}")
            return paper_chunk.id

        except Exception as e:
            logger.error(f"Failed to store chunk: {e}")
            raise DatabaseException(f"Failed to store chunk: {e}")

    async def store_chunks(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        embedded_chunks: list[EmbeddedChunk],
    ) -> StorageResult:
        """
        Store multiple embedded chunks in a batch.

        Args:
            session: Database session
            paper_id: ID of the parent paper
            embedded_chunks: List of chunks with embeddings

        Returns:
            StorageResult with stored chunk IDs and any errors
        """
        result = StorageResult()

        if not embedded_chunks:
            return result

        try:
            # Prepare all chunks for batch insert
            chunk_records = []
            for embedded in embedded_chunks:
                chunk = embedded.chunk
                chunk_records.append({
                    "id": uuid.uuid4(),
                    "paper_id": paper_id,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_index": chunk.chunk_index,
                    "embedding": embedded.embedding,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "chunk_metadata": chunk.chunk_metadata or {},
                })

            # Bulk insert
            stmt = pg_insert(PaperChunk).values(chunk_records)
            await session.execute(stmt)
            await session.flush()

            result.stored_count = len(chunk_records)
            result.chunk_ids = [r["id"] for r in chunk_records]

            logger.info(
                f"Stored {result.stored_count} chunks for paper {paper_id}"
            )

        except Exception as e:
            logger.error(f"Batch storage failed: {e}")
            result.failed_count = len(embedded_chunks)
            result.errors.append(str(e))

        return result

    async def delete_paper_chunks(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
    ) -> int:
        """
        Delete all chunks for a paper.

        Args:
            session: Database session
            paper_id: ID of the paper

        Returns:
            Number of deleted chunks
        """
        try:
            stmt = delete(PaperChunk).where(PaperChunk.paper_id == paper_id)
            result = await session.execute(stmt)
            await session.flush()

            deleted = result.rowcount
            logger.info(f"Deleted {deleted} chunks for paper {paper_id}")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise DatabaseException(f"Failed to delete chunks: {e}")

    # =========================================================================
    # Search Operations
    # =========================================================================

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

            logger.debug(
                f"Similarity search returned {len(search_results)} results"
            )
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

    async def search_by_text(
        self,
        session: AsyncSession,
        query_text: str,
        embedder,  # AsyncAzureEmbedder
        top_k: int = 10,
        **kwargs,
    ) -> list[SearchResult]:
        """
        Search by text query (embeds the query first).

        Args:
            session: Database session
            query_text: Text to search for
            embedder: AsyncAzureEmbedder instance
            top_k: Number of results to return
            **kwargs: Additional SearchQuery parameters

        Returns:
            List of SearchResult sorted by similarity
        """
        # Generate embedding for query
        query_embedding = await embedder.embed_single(query_text)

        # Perform search
        search_query = SearchQuery(
            embedding=query_embedding,
            top_k=top_k,
            **kwargs,
        )
        return await self.similarity_search(session, search_query)

    # =========================================================================
    # Index Management
    # =========================================================================

    async def create_ivfflat_index(
        self,
        session: AsyncSession,
        lists: int = 100,
    ) -> None:
        """
        Create IVFFlat index for approximate nearest neighbor search.

        Should be called after initial data ingestion for optimal performance.

        Args:
            session: Database session
            lists: Number of lists for IVFFlat (rule of thumb: sqrt(n) to n/1000)
        """
        try:
            # Drop existing index if any
            await session.execute(
                text("DROP INDEX IF EXISTS ix_paper_chunks_embedding_ivfflat")
            )

            # Create IVFFlat index with cosine distance
            await session.execute(
                text(f"""
                    CREATE INDEX ix_paper_chunks_embedding_ivfflat
                    ON paper_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {lists})
                """)
            )

            await session.commit()
            logger.info(f"Created IVFFlat index with {lists} lists")

        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to create IVFFlat index: {e}")
            raise DatabaseException(f"Failed to create IVFFlat index: {e}")

    async def create_hnsw_index(
        self,
        session: AsyncSession,
        m: int = 16,
        ef_construction: int = 64,
    ) -> None:
        """
        Create HNSW index for approximate nearest neighbor search.

        HNSW is generally faster than IVFFlat for queries but slower to build.

        Args:
            session: Database session
            m: Maximum number of connections per layer
            ef_construction: Size of dynamic candidate list for construction
        """
        try:
            # Drop existing index if any
            await session.execute(
                text("DROP INDEX IF EXISTS ix_paper_chunks_embedding_hnsw")
            )

            # Create HNSW index with cosine distance
            await session.execute(
                text(f"""
                    CREATE INDEX ix_paper_chunks_embedding_hnsw
                    ON paper_chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction})
                """)
            )

            await session.commit()
            logger.info(f"Created HNSW index with m={m}, ef_construction={ef_construction}")

        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to create HNSW index: {e}")
            raise DatabaseException(f"Failed to create HNSW index: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_chunk_count(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID | None = None,
    ) -> int:
        """Get total chunk count, optionally filtered by paper."""
        stmt = select(func.count(PaperChunk.id))
        if paper_id:
            stmt = stmt.where(PaperChunk.paper_id == paper_id)

        result = await session.execute(stmt)
        return result.scalar() or 0

    async def get_embedded_chunk_count(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID | None = None,
    ) -> int:
        """Get count of chunks that have embeddings."""
        stmt = select(func.count(PaperChunk.id)).where(
            PaperChunk.embedding.isnot(None)
        )
        if paper_id:
            stmt = stmt.where(PaperChunk.paper_id == paper_id)

        result = await session.execute(stmt)
        return result.scalar() or 0

    async def get_chunk_type_distribution(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID | None = None,
    ) -> dict[str, int]:
        """Get distribution of chunk types."""
        stmt = select(
            PaperChunk.chunk_type,
            func.count(PaperChunk.id).label("count"),
        ).group_by(PaperChunk.chunk_type)

        if paper_id:
            stmt = stmt.where(PaperChunk.paper_id == paper_id)

        result = await session.execute(stmt)
        return {row.chunk_type: row.count for row in result.fetchall()}


# Module-level factory functions


def get_vector_store(session: AsyncSession | None = None) -> VectorStore:
    """Create a VectorStore instance."""
    return VectorStore(session=session)

