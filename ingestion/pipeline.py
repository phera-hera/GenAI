"""
Ingestion Pipeline Orchestrator

Provides the complete pipeline for processing medical research papers:
1. Parse PDF using Azure Document Intelligence
2. Chunk into semantic sections
3. Generate embeddings using Azure OpenAI
4. Store in pgvector for retrieval

This module orchestrates all ingestion components and provides both
synchronous and asynchronous interfaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, BinaryIO

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseException, DocumentParsingError, LLMError
from app.db.models import Paper
from app.db.session import get_session_context

from .chunkers import ChunkedSection, SectionChunker
from .embedders import AsyncAzureEmbedder, EmbeddedChunk, EmbeddingResult, get_async_embedder
from .parsers import ParsedDocument, parse_pdf
from .storage import SearchQuery, SearchResult, StorageResult, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline."""

    # Chunking settings
    max_chunk_chars: int = 1200
    include_tables: bool = True

    # Embedding settings
    embedding_batch_size: int = 50
    skip_invalid_chunks: bool = True

    # Storage settings
    delete_existing_chunks: bool = True  # Delete old chunks before re-ingestion


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""

    paper_id: uuid.UUID
    paper_title: str | None
    gcp_path: str

    # Pipeline stages
    parsed: bool = False
    chunked: bool = False
    embedded: bool = False
    stored: bool = False

    # Statistics
    chunk_count: int = 0
    embedded_count: int = 0
    stored_count: int = 0
    failed_count: int = 0

    # Timing
    parse_time_ms: int = 0
    chunk_time_ms: int = 0
    embed_time_ms: int = 0
    store_time_ms: int = 0
    total_time_ms: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.stored and self.failed_count == 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Pipeline Result: {status}",
            f"  Paper: {self.paper_title or 'Unknown'} ({self.paper_id})",
            f"  GCP Path: {self.gcp_path}",
            f"  Chunks: {self.chunk_count} → Embedded: {self.embedded_count} → Stored: {self.stored_count}",
            f"  Failed: {self.failed_count}",
            f"  Time: {self.total_time_ms}ms (parse: {self.parse_time_ms}, chunk: {self.chunk_time_ms}, embed: {self.embed_time_ms}, store: {self.store_time_ms})",
        ]
        if self.errors:
            lines.append(f"  Errors: {', '.join(self.errors)}")
        return "\n".join(lines)


class IngestionPipeline:
    """
    Complete ingestion pipeline for medical research papers.

    Orchestrates parsing, chunking, embedding, and storage of PDFs
    into the vector database.

    Args:
        config: Pipeline configuration
        embedder: Optional AsyncAzureEmbedder (creates one if not provided)
        vector_store: Optional VectorStore (creates one if not provided)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedder: AsyncAzureEmbedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or PipelineConfig()
        self.embedder = embedder or get_async_embedder()
        self.vector_store = vector_store or get_vector_store()
        self.chunker = SectionChunker(
            max_chunk_chars=self.config.max_chunk_chars,
            include_tables=self.config.include_tables,
        )

    def compute_file_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content for deduplication."""
        return hashlib.sha256(content).hexdigest()

    async def check_duplicate(
        self,
        session: AsyncSession,
        file_hash: str,
    ) -> Paper | None:
        """
        Check if a paper with the same hash already exists.

        Args:
            session: Database session
            file_hash: SHA-256 hash of the PDF

        Returns:
            Existing Paper if duplicate found, None otherwise
        """
        stmt = select(Paper).where(Paper.file_hash == file_hash)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def process_paper(
        self,
        session: AsyncSession,
        pdf_content: bytes,
        gcp_path: str,
        paper_id: uuid.UUID | None = None,
        skip_duplicate_check: bool = False,
        progress_callback: callable | None = None,
    ) -> PipelineResult:
        """
        Process a PDF through the complete ingestion pipeline.

        Args:
            session: Database session
            pdf_content: Raw PDF bytes
            gcp_path: GCP Cloud Storage path to the PDF
            paper_id: Optional existing paper ID (for re-processing)
            skip_duplicate_check: Skip hash-based deduplication
            progress_callback: Optional callback(stage, progress) for updates

        Returns:
            PipelineResult with processing details
        """
        import time

        start_time = time.monotonic()

        # Initialize result
        result = PipelineResult(
            paper_id=paper_id or uuid.uuid4(),
            paper_title=None,
            gcp_path=gcp_path,
        )

        file_hash = self.compute_file_hash(pdf_content)

        try:
            # Check for duplicate
            if not skip_duplicate_check:
                existing = await self.check_duplicate(session, file_hash)
                if existing:
                    result.paper_id = existing.id
                    result.paper_title = existing.title
                    result.errors.append(f"Duplicate paper found: {existing.id}")
                    return result

            # Stage 1: Parse PDF
            if progress_callback:
                progress_callback("parse", 0)

            parse_start = time.monotonic()
            try:
                parsed = parse_pdf(pdf_content, source_file=gcp_path)
                result.parsed = True
            except Exception as e:
                logger.error(f"PDF parsing failed: {e}")
                result.errors.append(f"Parse error: {e}")
                return result
            finally:
                result.parse_time_ms = int((time.monotonic() - parse_start) * 1000)

            result.paper_title = parsed.metadata.title

            if progress_callback:
                progress_callback("parse", 100)

            # Stage 2: Chunk document
            if progress_callback:
                progress_callback("chunk", 0)

            chunk_start = time.monotonic()
            try:
                chunks = self.chunker.chunk_document(parsed)
                result.chunked = True
                result.chunk_count = len(chunks)
            except Exception as e:
                logger.error(f"Chunking failed: {e}")
                result.errors.append(f"Chunk error: {e}")
                return result
            finally:
                result.chunk_time_ms = int((time.monotonic() - chunk_start) * 1000)

            if progress_callback:
                progress_callback("chunk", 100)

            if not chunks:
                result.errors.append("No chunks generated from document")
                return result

            # Stage 3: Generate embeddings
            if progress_callback:
                progress_callback("embed", 0)

            embed_start = time.monotonic()
            try:

                def embed_progress(done: int, total: int) -> None:
                    if progress_callback:
                        progress_callback("embed", int(done / total * 100))

                embed_result = await self.embedder.embed_chunks(
                    chunks,
                    skip_invalid=self.config.skip_invalid_chunks,
                    progress_callback=embed_progress,
                )
                result.embedded = True
                result.embedded_count = embed_result.success_count
                result.failed_count = embed_result.failure_count

            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                result.errors.append(f"Embed error: {e}")
                return result
            finally:
                result.embed_time_ms = int((time.monotonic() - embed_start) * 1000)

            if not embed_result.embedded_chunks:
                result.errors.append("No chunks were successfully embedded")
                return result

            # Stage 4: Store in database
            if progress_callback:
                progress_callback("store", 0)

            store_start = time.monotonic()
            try:
                # Create or update paper record
                paper = await self._upsert_paper(
                    session=session,
                    paper_id=result.paper_id,
                    parsed=parsed,
                    gcp_path=gcp_path,
                    file_hash=file_hash,
                )
                result.paper_id = paper.id

                # Delete existing chunks if re-processing
                if self.config.delete_existing_chunks:
                    await self.vector_store.delete_paper_chunks(session, paper.id)

                # Store new chunks
                storage_result = await self.vector_store.store_chunks(
                    session=session,
                    paper_id=paper.id,
                    embedded_chunks=embed_result.embedded_chunks,
                )
                result.stored = True
                result.stored_count = storage_result.stored_count

                if storage_result.errors:
                    result.errors.extend(storage_result.errors)

                # Mark paper as processed
                paper.is_processed = True
                paper.processed_at = datetime.utcnow()
                await session.flush()

            except Exception as e:
                logger.error(f"Storage failed: {e}")
                result.errors.append(f"Store error: {e}")
                return result
            finally:
                result.store_time_ms = int((time.monotonic() - store_start) * 1000)

            if progress_callback:
                progress_callback("store", 100)

        finally:
            result.total_time_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(result.summary())
        return result

    async def _upsert_paper(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        parsed: ParsedDocument,
        gcp_path: str,
        file_hash: str,
    ) -> Paper:
        """Create or update a paper record."""
        # Check if paper exists
        stmt = select(Paper).where(Paper.id == paper_id)
        result = await session.execute(stmt)
        paper = result.scalar_one_or_none()

        metadata = parsed.metadata

        if paper:
            # Update existing
            paper.title = metadata.title or paper.title
            paper.authors = ", ".join(metadata.authors) if metadata.authors else paper.authors
            paper.journal = metadata.journal or paper.journal
            paper.publication_year = metadata.publication_year or paper.publication_year
            paper.doi = metadata.doi or paper.doi
            paper.abstract = parsed.get_abstract() or paper.abstract
            paper.file_hash = file_hash
            paper.gcp_path = gcp_path
        else:
            # Create new
            paper = Paper(
                id=paper_id,
                title=metadata.title or "Untitled",
                authors=", ".join(metadata.authors) if metadata.authors else None,
                journal=metadata.journal,
                publication_year=metadata.publication_year,
                doi=metadata.doi,
                abstract=parsed.get_abstract(),
                gcp_path=gcp_path,
                file_hash=file_hash,
                is_processed=False,
            )
            session.add(paper)

        await session.flush()
        return paper

    async def reprocess_paper(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        pdf_content: bytes,
    ) -> PipelineResult:
        """
        Reprocess an existing paper (e.g., after chunking strategy change).

        Args:
            session: Database session
            paper_id: ID of the existing paper
            pdf_content: Raw PDF bytes

        Returns:
            PipelineResult with processing details
        """
        # Get existing paper
        stmt = select(Paper).where(Paper.id == paper_id)
        result = await session.execute(stmt)
        paper = result.scalar_one_or_none()

        if not paper:
            raise DatabaseException(f"Paper not found: {paper_id}")

        return await self.process_paper(
            session=session,
            pdf_content=pdf_content,
            gcp_path=paper.gcp_path,
            paper_id=paper_id,
            skip_duplicate_check=True,
        )


# Convenience functions for common operations


async def process_pdf(
    pdf_content: bytes,
    gcp_path: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Process a PDF through the complete ingestion pipeline.

    Convenience function that handles session management.

    Args:
        pdf_content: Raw PDF bytes
        gcp_path: GCP Cloud Storage path to the PDF
        config: Optional pipeline configuration

    Returns:
        PipelineResult with processing details
    """
    pipeline = IngestionPipeline(config=config)

    async with get_session_context() as session:
        return await pipeline.process_paper(
            session=session,
            pdf_content=pdf_content,
            gcp_path=gcp_path,
        )


async def search_similar(
    query_text: str,
    top_k: int = 10,
    paper_ids: list[uuid.UUID] | None = None,
    chunk_types: list[str] | None = None,
) -> list[SearchResult]:
    """
    Search for similar chunks using text query.

    Convenience function that handles embedding and session management.

    Args:
        query_text: Text to search for
        top_k: Number of results to return
        paper_ids: Optional filter by paper IDs
        chunk_types: Optional filter by chunk types

    Returns:
        List of SearchResult sorted by similarity
    """
    from app.db.models import ChunkType

    embedder = get_async_embedder()
    vector_store = get_vector_store()

    # Convert string chunk types to enum
    chunk_type_enums = None
    if chunk_types:
        chunk_type_enums = [ChunkType(ct) for ct in chunk_types]

    async with get_session_context() as session:
        return await vector_store.search_by_text(
            session=session,
            query_text=query_text,
            embedder=embedder,
            top_k=top_k,
            paper_ids=paper_ids,
            chunk_types=chunk_type_enums,
        )

