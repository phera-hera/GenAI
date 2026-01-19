"""
Docling Hierarchical Chunker for Medical Research Papers

Uses Docling's native hierarchical structure to create chunks that respect
semantic boundaries and section hierarchy. Implements configurable overlap
to maintain context across chunk boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from medical_agent.core.config import settings
from medical_agent.infrastructure.database.models import ChunkType
from medical_agent.ingestion.parsers import (
    DocumentSection,
    ExtractedTable,
    ParsedDocument,
    SectionType,
)

logger = logging.getLogger(__name__)


# Maps parser section types to chunk types stored in the database.
SECTION_TO_CHUNK_TYPE: dict[SectionType, ChunkType] = {
    SectionType.ABSTRACT: ChunkType.ABSTRACT,
    SectionType.INTRODUCTION: ChunkType.INTRODUCTION,
    SectionType.BACKGROUND: ChunkType.INTRODUCTION,
    SectionType.METHODS: ChunkType.METHODS,
    SectionType.RESULTS: ChunkType.RESULTS,
    SectionType.DISCUSSION: ChunkType.DISCUSSION,
    SectionType.CONCLUSION: ChunkType.CONCLUSION,
    SectionType.REFERENCES: ChunkType.REFERENCES,
    SectionType.ACKNOWLEDGMENTS: ChunkType.OTHER,
    SectionType.APPENDIX: ChunkType.OTHER,
    SectionType.OTHER: ChunkType.OTHER,
}


@dataclass
class ChunkedSection:
    """Lightweight representation of a chunk ready for embedding/storage."""

    content: str
    chunk_type: ChunkType
    chunk_index: int
    section_title: str | None = None
    page_number: int | None = None
    chunk_metadata: dict[str, Any] = field(default_factory=dict)


class DoclingHierarchicalChunker:
    """
    Hierarchical chunker that uses Docling's structure for semantic chunking.

    Features:
    - Respects section boundaries (no mid-concept splits)
    - Configurable overlap between chunks for context preservation
    - Hierarchical section awareness
    - Compatible with existing pipeline (returns ChunkedSection list)

    Args:
        max_chunk_chars: Maximum characters per chunk
        overlap_chars: Number of characters to overlap between chunks
        include_tables: Whether to emit table chunks
        respect_section_boundaries: Don't split across section boundaries
    """

    def __init__(
        self,
        max_chunk_chars: int = 1200,
        overlap_chars: int = 200,
        include_tables: bool = True,
        respect_section_boundaries: bool = True,
    ):
        """
        Initialize the Docling hierarchical chunker.

        Args:
            max_chunk_chars: Maximum characters per chunk
            overlap_chars: Characters to overlap between chunks
            include_tables: Whether to create table chunks
            respect_section_boundaries: Don't split sections mid-concept
        """
        if max_chunk_chars < 200:
            raise ValueError("max_chunk_chars must be at least 200 characters")
        if overlap_chars >= max_chunk_chars:
            raise ValueError("overlap_chars must be less than max_chunk_chars")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")

        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.include_tables = include_tables
        self.respect_section_boundaries = respect_section_boundaries

    def chunk_document(self, parsed: ParsedDocument) -> list[ChunkedSection]:
        """
        Create ordered chunks from a parsed document.

        Respects Docling's hierarchical structure and section boundaries.

        Args:
            parsed: Parsed document from Docling or other parser

        Returns:
            List of ChunkedSection objects ready for embedding
        """
        chunks: list[ChunkedSection] = []
        chunk_index = 0

        # Level 1: metadata + abstract
        metadata_chunk = self._build_metadata_abstract_chunk(parsed, chunk_index)
        if metadata_chunk:
            chunks.append(metadata_chunk)
            chunk_index += 1

        # Level 2: hierarchical section chunks (skip abstract, covered above)
        for section in parsed.sections:
            if section.section_type == SectionType.ABSTRACT:
                continue

            section_chunks = self._chunk_section_hierarchical(section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Level 3: tables
        if self.include_tables:
            table_chunks = self._chunk_tables(parsed.tables, chunk_index)
            chunks.extend(table_chunks)

        logger.info(
            f"Docling hierarchical chunking complete: {len(chunks)} chunks "
            f"(max: {self.max_chunk_chars} chars, overlap: {self.overlap_chars} chars)"
        )

        return chunks

    def _build_metadata_abstract_chunk(
        self,
        parsed: ParsedDocument,
        chunk_index: int,
    ) -> ChunkedSection | None:
        """Combine paper metadata and abstract into a single chunk."""
        lines: list[str] = []
        meta = parsed.metadata

        if meta.title:
            lines.append(f"Title: {meta.title}")
        if meta.authors:
            lines.append(f"Authors: {', '.join(meta.authors)}")
        if meta.journal:
            lines.append(f"Journal: {meta.journal}")
        if meta.publication_year:
            lines.append(f"Year: {meta.publication_year}")
        if meta.doi:
            lines.append(f"DOI: {meta.doi}")

        abstract = parsed.get_abstract()
        if abstract:
            lines.append("Abstract:")
            lines.append(abstract.strip())

        # Guard against empty chunk
        content = "\n".join(lines).strip()
        if not content:
            return None

        return ChunkedSection(
            content=content,
            chunk_type=ChunkType.ABSTRACT,
            chunk_index=chunk_index,
            section_title="Abstract",
            chunk_metadata={"source": "metadata_and_abstract"},
        )

    def _chunk_section_hierarchical(
        self,
        section: DocumentSection,
        start_index: int,
    ) -> list[ChunkedSection]:
        """
        Split a section into hierarchical, overlapping chunks.

        Respects section boundaries and creates overlap between chunks
        to maintain context.

        Args:
            section: Document section to chunk
            start_index: Starting chunk index

        Returns:
            List of ChunkedSection objects
        """
        text = (section.full_content or section.content or "").strip()
        if not text:
            return []

        # Split into paragraphs as semantic units
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks: list[ChunkedSection] = []
        current_chunk_paragraphs: list[str] = []
        overlap_buffer: list[str] = []  # For maintaining overlap

        def create_chunk(paragraphs_list: list[str]) -> ChunkedSection | None:
            """Helper to create a chunk from paragraphs."""
            if not paragraphs_list:
                return None

            content = "\n\n".join(paragraphs_list).strip()
            if not content:
                return None

            chunk = ChunkedSection(
                content=content,
                chunk_type=SECTION_TO_CHUNK_TYPE.get(
                    section.section_type, ChunkType.OTHER
                ),
                chunk_index=start_index + len(chunks),
                section_title=section.title,
                page_number=min(section.page_numbers) if section.page_numbers else None,
                chunk_metadata={
                    "section_type": section.section_type.value,
                    "hierarchical": True,
                },
            )
            return chunk

        def get_overlap_paragraphs(paragraphs_list: list[str]) -> list[str]:
            """Extract paragraphs for overlap based on char count."""
            if not paragraphs_list or self.overlap_chars == 0:
                return []

            overlap_paras = []
            char_count = 0

            # Work backwards from end of chunk
            for para in reversed(paragraphs_list):
                if char_count + len(para) <= self.overlap_chars:
                    overlap_paras.insert(0, para)
                    char_count += len(para) + 2  # +2 for \n\n
                else:
                    break

            return overlap_paras

        # Process paragraphs
        for paragraph in paragraphs:
            # Try adding paragraph to current chunk
            candidate_paragraphs = current_chunk_paragraphs + [paragraph]
            candidate_text = "\n\n".join(candidate_paragraphs)

            if len(candidate_text) <= self.max_chunk_chars:
                # Fits in current chunk
                current_chunk_paragraphs.append(paragraph)
            else:
                # Current chunk is full, flush it
                if current_chunk_paragraphs:
                    chunk = create_chunk(current_chunk_paragraphs)
                    if chunk:
                        chunks.append(chunk)

                    # Extract overlap paragraphs for next chunk
                    overlap_buffer = get_overlap_paragraphs(current_chunk_paragraphs)

                # Start new chunk with overlap + current paragraph
                current_chunk_paragraphs = overlap_buffer + [paragraph]
                overlap_buffer = []

        # Flush remaining paragraphs
        if current_chunk_paragraphs:
            chunk = create_chunk(current_chunk_paragraphs)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _chunk_tables(
        self,
        tables: list[ExtractedTable],
        start_index: int,
    ) -> list[ChunkedSection]:
        """Create dedicated chunks for tables."""
        chunks: list[ChunkedSection] = []

        for table in tables:
            markdown = table.to_markdown()
            if not markdown:
                continue

            content = f"Table {table.table_id}:\n{markdown}"

            # Add caption if present
            if table.caption:
                content = f"{table.caption}\n\n{content}"

            chunk = ChunkedSection(
                content=content,
                chunk_type=ChunkType.TABLE,
                chunk_index=start_index + len(chunks),
                section_title=table.caption or f"Table {table.table_id}",
                page_number=table.page_number,
                chunk_metadata={
                    "table_id": table.table_id,
                    "page_number": table.page_number,
                    "caption": table.caption,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                },
            )
            chunks.append(chunk)

        return chunks


# Convenience function to create chunker with settings
def get_docling_chunker() -> DoclingHierarchicalChunker:
    """Create a Docling chunker with settings from config."""
    return DoclingHierarchicalChunker(
        max_chunk_chars=1200,  # Default from existing chunker
        overlap_chars=getattr(settings, "chunk_overlap_chars", 200),
        include_tables=True,
        respect_section_boundaries=getattr(settings, "respect_section_boundaries", True),
    )
