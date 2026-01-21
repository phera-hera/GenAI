"""
LlamaIndex-Based Ingestion Pipeline

Modern ingestion pipeline using LlamaIndex components to:
1. Parse PDFs with DoclingReader (JSON export for structured tables)
2. Chunk with DoclingNodeParser (section-aware)
3. Extract medical metadata (8 fields)
4. Generate embeddings (Azure OpenAI)
5. Store in pgvector

This pipeline fixes table extraction issues by using JSON export instead of
markdown, while maintaining compatibility with existing storage and retrieval.
"""

import hashlib
import io
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.readers.docling import DoclingReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.config import settings
from medical_agent.core.exceptions import DatabaseException
from medical_agent.infrastructure.database.models import Paper
from medical_agent.infrastructure.database.session import get_session_context
from medical_agent.ingestion.metadata.llamaindex_extractor import (
    LlamaIndexMedicalMetadataExtractor,
)
from medical_agent.ingestion.storage.llamaindex_vector_store import (
    MedicalPGVectorStore,
)

logger = logging.getLogger(__name__)


@dataclass
class LlamaIndexPipelineConfig:
    """Configuration for the LlamaIndex ingestion pipeline."""

    # Docling reader settings
    docling_export_type: str = "json"  # Use JSON for structured tables
    docling_enable_ocr: bool = True  # Auto OCR for scanned papers
    docling_table_structure_mode: str = "accurate"  # TableFormer accurate mode

    # Metadata extraction settings
    extract_metadata: bool = True
    extract_table_summaries: bool = True

    # Storage settings
    delete_existing_chunks: bool = True  # Delete old chunks before re-ingestion

    # Chunking settings
    chunk_size: int = 1200  # Match existing pipeline max_chunk_chars
    chunk_overlap: int = 0  # No overlap - sections are semantically bounded

    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072


@dataclass
class LlamaIndexPipelineResult:
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


class LlamaIndexIngestionPipeline:
    """
    LlamaIndex-based ingestion pipeline for medical papers.

    Pipeline stages:
    1. Parse PDF with DoclingReader (JSON export for structured tables)
    2. Chunk with DoclingNodeParser (section-aware)
    3. Extract medical metadata (8 fields)
    4. Generate embeddings (Azure OpenAI)
    5. Store in pgvector

    Args:
        config: Pipeline configuration
        vector_store: Optional MedicalPGVectorStore instance
    """

    def __init__(
        self,
        config: LlamaIndexPipelineConfig | None = None,
        vector_store: MedicalPGVectorStore | None = None,
    ):
        self.config = config or LlamaIndexPipelineConfig()
        self.vector_store = vector_store or MedicalPGVectorStore()

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
        self.metadata_extractor = LlamaIndexMedicalMetadataExtractor(
            extract_table_summaries=self.config.extract_table_summaries,
        )

        # Initialize node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

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
    ) -> LlamaIndexPipelineResult:
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
            LlamaIndexPipelineResult with processing details
        """
        start_time = time.monotonic()

        # Initialize result
        result = LlamaIndexPipelineResult(
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
                # DoclingReader expects a file-like object
                pdf_file = io.BytesIO(pdf_content)
                documents = self.reader.load_data(file=pdf_file)
                result.parsed = True

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
                nodes = await self.pipeline.arun(documents=[document])
                result.chunked = True
                result.embedded = True
                result.node_count = len(nodes)
                result.chunk_count = len(nodes)

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
                paper = await self._upsert_paper(
                    session=session,
                    paper_id=result.paper_id,
                    document=document,
                    gcp_path=gcp_path,
                    file_hash=file_hash,
                )
                result.paper_id = paper.id
                result.paper_title = paper.title

                # Delete existing chunks if re-processing
                if self.config.delete_existing_chunks:
                    await self.vector_store.delete_paper_chunks(paper.id)

                # Store nodes
                node_ids = await self.vector_store.add_nodes(
                    nodes=nodes,
                    paper_id=paper.id,
                )
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

    async def _upsert_paper(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        document: Document,
        gcp_path: str,
        file_hash: str,
    ) -> Paper:
        """Create or update a paper record."""
        # Check if paper exists
        stmt = select(Paper).where(Paper.id == paper_id)
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
    ) -> LlamaIndexPipelineResult:
        """
        Reprocess an existing paper.

        Args:
            session: Database session
            paper_id: ID of the existing paper
            pdf_content: Raw PDF bytes

        Returns:
            LlamaIndexPipelineResult with processing details
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
    config: LlamaIndexPipelineConfig | None = None,
) -> LlamaIndexPipelineResult:
    """
    Process a PDF through the LlamaIndex ingestion pipeline.

    Convenience function that handles session management.

    Args:
        pdf_content: Raw PDF bytes
        gcp_path: GCP Cloud Storage path to the PDF
        config: Optional pipeline configuration

    Returns:
        LlamaIndexPipelineResult with processing details
    """
    pipeline = LlamaIndexIngestionPipeline(config=config)

    async with get_session_context() as session:
        return await pipeline.process_paper(
            session=session,
            pdf_content=pdf_content,
            gcp_path=gcp_path,
        )
