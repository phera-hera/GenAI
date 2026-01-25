"""
LlamaIndex-Based Ingestion Pipeline

Modern ingestion pipeline using LlamaIndex components to:
1. Parse PDFs with DoclingReader (JSON export for structured tables AND sections)
2. Chunk with DoclingNodeParser + HybridChunker (section-aware + token-limited)
3. Extract medical metadata (8 fields)
4. Generate embeddings (Azure OpenAI)
5. Store in pgvector

HybridChunker combines hierarchical structure (sections, paragraphs, tables) with
token awareness (max 512 tokens per chunk) to prevent embedding truncation while
preserving document structure.
"""

import hashlib
import io
import logging
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from docling.chunking import HybridChunker
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.config import settings
from medical_agent.core.exceptions import DatabaseException
from medical_agent.infrastructure.database.models import Paper, PaperChunk
from medical_agent.infrastructure.database.session import get_session_context
from medical_agent.ingestion.metadata.extractor import MedicalMetadataExtractor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the LlamaIndex ingestion pipeline."""

    # Docling reader settings
    docling_export_type: str = "json"  # Use JSON for structured tables AND section metadata
    docling_enable_ocr: bool = True  # Auto OCR for scanned papers
    docling_table_structure_mode: str = "accurate"  # TableFormer accurate mode

    # Metadata extraction settings
    extract_metadata: bool = True

    # Storage settings
    delete_existing_chunks: bool = True  # Delete old chunks before re-ingestion

    # Chunking settings for HybridChunker
    max_chunk_tokens: int = 512   # Maximum tokens per chunk (prevents embedding truncation)
    min_chunk_tokens: int = 64    # Minimum tokens per chunk (avoids tiny chunks)
    merge_small_chunks: bool = True  # Merge small adjacent chunks

    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072


@dataclass
class PipelineResult:
    """Result of a complete LlamaIndex pipeline run."""

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
    node_count: int = 0
    embedded_count: int = 0  # For backward compatibility with script
    stored_count: int = 0
    failed_count: int = 0

    # Timing
    parse_time_ms: int = 0
    pipeline_time_ms: int = 0
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
            f"LlamaIndex Pipeline Result: {status}",
            f"  Paper: {self.paper_title or 'Unknown'} ({self.paper_id})",
            f"  GCP Path: {self.gcp_path}",
            f"  Nodes: {self.node_count} → Stored: {self.stored_count}",
            f"  Failed: {self.failed_count}",
            f"  Time: {self.total_time_ms}ms (parse: {self.parse_time_ms}, pipeline: {self.pipeline_time_ms}, store: {self.store_time_ms})",
        ]
        if self.errors:
            lines.append(f"  Errors: {', '.join(self.errors)}")
        return "\n".join(lines)


class MedicalIngestionPipeline:
    """
    LlamaIndex-based ingestion pipeline for medical papers.

    Pipeline stages:
    1. Parse PDF with DoclingReader (JSON export for structured tables)
    2. Chunk with DoclingNodeParser (section-aware)
    3. Extract medical metadata (8 fields)
    4. Generate embeddings (Azure OpenAI)
    5. Store in pgvector (native PGVectorStore with hybrid_search=True)

    Args:
        config: Pipeline configuration
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()

        # Initialize native PGVectorStore with hybrid search support
        # Table name: LlamaIndex prefixes with "data_", so "paper_chunks" → "data_paper_chunks"
        self.vector_store = PGVectorStore.from_params(
            database=settings.postgres_db,
            host=settings.postgres_host,
            password=settings.postgres_password,
            port=settings.postgres_port,
            user=settings.postgres_user,
            table_name="paper_chunks",  # Will use data_paper_chunks (LlamaIndex adds "data_" prefix)
            embed_dim=settings.embedding_dimension,
            hybrid_search=True,  # Enable built-in BM25 + vector fusion
            text_search_config="english",  # Full-text search language
            perform_setup=False,  # Use existing table schema (don't create new table)
        )

        # Initialize Docling reader with optimal settings
        self.reader = DoclingReader(
            export_type=self.config.docling_export_type,
            ocr=self.config.docling_enable_ocr,
            table_structure_mode=self.config.docling_table_structure_mode,
        )

        # Initialize embedding model
        # Determine which credentials to use
        if settings.azure_openai_embedding_api_key and settings.azure_openai_embedding_endpoint:
            # Use separate embedding credentials
            api_key = settings.azure_openai_embedding_api_key
            azure_endpoint = settings.azure_openai_embedding_endpoint
            api_version = settings.azure_openai_embedding_api_version
            deployment_name = settings.azure_openai_embedding_deployment_name
        else:
            # Fall back to main credentials
            api_key = settings.azure_openai_api_key
            azure_endpoint = settings.azure_openai_endpoint
            api_version = settings.azure_openai_api_version
            deployment_name = settings.azure_openai_embedding_deployment_name

        self.embedding_model = AzureOpenAIEmbedding(
            model=self.config.embedding_model,
            deployment_name=deployment_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            dimensions=self.config.embedding_dimensions,
        )

        # Initialize metadata extractor
        self.metadata_extractor = MedicalMetadataExtractor()

        # Initialize HybridChunker for token-aware section chunking
        # Use default tokenizer (HybridChunker has its own tokenizer handling)
        hybrid_chunker = HybridChunker(
            max_tokens=self.config.max_chunk_tokens,  # Prevent embedding truncation
            min_tokens=self.config.min_chunk_tokens,  # Avoid tiny chunks
            merge_peers=self.config.merge_small_chunks,  # Merge small adjacent chunks
            heading_as_metadata=True,  # Preserve section hierarchy
        )

        # Initialize DoclingNodeParser with HybridChunker
        # Splits Docling JSON by structure + enforces token limits
        self.node_parser = DoclingNodeParser(chunker=hybrid_chunker)

        # Build ingestion pipeline
        self.pipeline = IngestionPipeline(
            transformations=[
                self.node_parser,  # Chunk documents into nodes
                self.metadata_extractor,  # Medical metadata extraction
                self.embedding_model,  # Embedding generation
            ],
        )

        logger.info("Initialized LlamaIndex ingestion pipeline")

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
        progress_callback: Callable | None = None,
    ) -> PipelineResult:
        """
        Process a PDF through the LlamaIndex ingestion pipeline.

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

            # Stage 1: Parse PDF with DoclingReader
            if progress_callback:
                progress_callback("parse", 0)

            parse_start = time.monotonic()
            try:
                # DoclingReader expects a file path, so write to temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                    tmp_file.write(pdf_content)
                    tmp_path = tmp_file.name

                try:
                    logger.info(f"Starting PDF parse for {gcp_path} ({len(pdf_content)} bytes)")
                    documents = list(self.reader.lazy_load_data(file_path=tmp_path))
                    logger.info(f"PDF parse completed: extracted {len(documents)} documents")
                    result.parsed = True
                finally:
                    # Clean up temp file
                    Path(tmp_path).unlink(missing_ok=True)

                if not documents:
                    result.errors.append("No documents extracted from PDF")
                    return result

                # Get the first document (DoclingReader typically returns one document per PDF)
                document = documents[0]
                result.paper_title = document.metadata.get("title", "Unknown")

                # Add paper metadata to document
                document.metadata["paper_id"] = str(result.paper_id)
                document.metadata["gcp_path"] = gcp_path
                document.id_ = str(result.paper_id)  # Use paper_id as document ID

            except Exception as e:
                logger.error(f"PDF parsing failed: {e}", exc_info=True)
                result.errors.append(f"Parse error: {e}")
                return result
            finally:
                result.parse_time_ms = int((time.monotonic() - parse_start) * 1000)

            if progress_callback:
                progress_callback("parse", 100)

            # Stage 2-4: Run ingestion pipeline (chunk, extract metadata, embed)
            if progress_callback:
                progress_callback("pipeline", 0)

            pipeline_start = time.monotonic()
            try:
                # Run the complete pipeline
                logger.info(f"Starting pipeline (chunk + embed) for paper {result.paper_id}")
                nodes = await self.pipeline.arun(documents=[document])
                logger.info(f"Pipeline completed: generated {len(nodes)} nodes")
                result.chunked = True
                result.embedded = True
                result.node_count = len(nodes)
                result.chunk_count = len(nodes)
                result.embedded_count = len(nodes)

                logger.info(f"Pipeline generated {len(nodes)} nodes for paper {result.paper_id}")

            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}", exc_info=True)
                result.errors.append(f"Pipeline error: {e}")
                return result
            finally:
                result.pipeline_time_ms = int((time.monotonic() - pipeline_start) * 1000)

            if progress_callback:
                progress_callback("pipeline", 100)

            if not nodes:
                result.errors.append("No nodes generated from pipeline")
                return result

            # Stage 5: Store in database
            if progress_callback:
                progress_callback("store", 0)

            store_start = time.monotonic()
            try:
                # Create or update paper record
                logger.info(f"Upserting paper record for {result.paper_id}")
                paper = await self._upsert_paper(
                    session=session,
                    paper_id=result.paper_id,
                    document=document,
                    gcp_path=gcp_path,
                    file_hash=file_hash,
                )
                result.paper_id = paper.id
                result.paper_title = paper.title
                logger.info(f"Paper record created: {paper.title}")

                # Delete existing chunks if re-processing
                if self.config.delete_existing_chunks:
                    logger.info(f"Deleting existing chunks for paper {paper.id}")
                    deleted_count = await self._delete_paper_chunks(session, paper.id)
                    logger.info(f"Deleted {deleted_count} existing chunks")

                # Add paper_id to all nodes' metadata
                for node in nodes:
                    if node.metadata is None:
                        node.metadata = {}
                    node.metadata["paper_id"] = str(paper.id)

                # Store nodes using native PGVectorStore
                logger.info(f"Starting to store {len(nodes)} nodes to vector store")
                node_ids = self.vector_store.add(nodes)
                logger.info(f"Successfully stored {len(node_ids)} nodes")
                result.stored = True
                result.stored_count = len(node_ids)

                # Mark paper as processed
                paper.is_processed = True
                paper.processed_at = datetime.utcnow()
                await session.flush()

            except Exception as e:
                logger.error(f"Storage failed: {e}", exc_info=True)
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

    async def _delete_paper_chunks(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
    ) -> int:
        """
        Delete all chunks for a paper by querying metadata_ JSONB.

        Args:
            session: Database session
            paper_id: UUID of the paper

        Returns:
            Number of deleted chunks
        """
        from sqlalchemy import delete

        try:
            # Query by metadata_->>'paper_id' since paper_id is now in JSONB
            stmt = delete(PaperChunk).where(
                PaperChunk.metadata_['paper_id'].astext == str(paper_id)
            )
            result = await session.execute(stmt)
            deleted_count = result.rowcount
            await session.commit()
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}", exc_info=True)
            raise DatabaseException(f"Failed to delete chunks: {e}")

    async def _upsert_paper(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        document: Document,
        gcp_path: str,
        file_hash: str,
    ) -> Paper:
        """Create or update a paper record."""
        # Check if paper exists by ID first
        stmt = select(Paper).where(Paper.id == paper_id)
        result = await session.execute(stmt)
        paper = result.scalar_one_or_none()

        # If not found by ID, check by file_hash (for re-processing with new ID)
        if not paper:
            stmt = select(Paper).where(Paper.file_hash == file_hash)
            result = await session.execute(stmt)
            paper = result.scalar_one_or_none()

        # Extract metadata from document
        metadata = document.metadata or {}
        title = metadata.get("title", "Untitled")
        authors = metadata.get("authors")
        if isinstance(authors, list):
            authors = ", ".join(authors)

        abstract = metadata.get("abstract")
        journal = metadata.get("journal")
        publication_year = metadata.get("publication_year")
        doi = metadata.get("doi")

        if paper:
            # Update existing
            paper.title = title or paper.title
            paper.authors = authors or paper.authors
            paper.journal = journal or paper.journal
            paper.publication_year = publication_year or paper.publication_year
            paper.doi = doi or paper.doi
            paper.abstract = abstract or paper.abstract
            paper.file_hash = file_hash
            paper.gcp_path = gcp_path
        else:
            # Create new
            paper = Paper(
                id=paper_id,
                title=title,
                authors=authors,
                journal=journal,
                publication_year=publication_year,
                doi=doi,
                abstract=abstract,
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
        Reprocess an existing paper.

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


# Convenience functions


async def process_pdf_llamaindex(
    pdf_content: bytes,
    gcp_path: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Process a PDF through the LlamaIndex ingestion pipeline.

    Convenience function that handles session management.

    Args:
        pdf_content: Raw PDF bytes
        gcp_path: GCP Cloud Storage path to the PDF
        config: Optional pipeline configuration

    Returns:
        PipelineResult with processing details
    """
    pipeline = MedicalIngestionPipeline(config=config)

    async with get_session_context() as session:
        return await pipeline.process_paper(
            session=session,
            pdf_content=pdf_content,
            gcp_path=gcp_path,
        )
