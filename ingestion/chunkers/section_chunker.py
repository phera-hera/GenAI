"""
Section-based chunking for medical research papers.

Takes a `ParsedDocument` produced by the ingestion parsers and returns
lightweight chunk objects that can be embedded and stored as `PaperChunk`
records. The chunking strategy mirrors the plan:

- Level 1: metadata + abstract as a single, high-priority chunk
- Level 2: individual sections, split into manageable pieces
- Level 3: tables preserved separately with markdown representation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.db.models import ChunkType
from ingestion.parsers import (
    DocumentSection,
    ExtractedTable,
    ParsedDocument,
    SectionType,
)

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


class SectionChunker:
    """
    Chunk a parsed medical paper into semantic, storage-ready pieces.

    Args:
        max_chunk_chars: Maximum characters allowed per section chunk. Content
            is split on paragraph boundaries to stay within this budget.
        include_tables: Whether to emit table chunks.
    """

    def __init__(self, max_chunk_chars: int = 1200, include_tables: bool = True):
        if max_chunk_chars < 200:
            raise ValueError("max_chunk_chars must be at least 200 characters")
        self.max_chunk_chars = max_chunk_chars
        self.include_tables = include_tables

    def chunk_document(self, parsed: ParsedDocument) -> list[ChunkedSection]:
        """Create ordered chunks from a parsed document."""
        chunks: list[ChunkedSection] = []
        chunk_index = 0

        # Level 1: metadata + abstract
        metadata_chunk = self._build_metadata_abstract_chunk(parsed, chunk_index)
        if metadata_chunk:
            chunks.append(metadata_chunk)
            chunk_index += 1

        # Level 2: section chunks (skip abstract, covered above)
        for section in parsed.sections:
            if section.section_type == SectionType.ABSTRACT:
                continue
            section_chunks = self._chunk_section(section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Level 3: tables
        if self.include_tables:
            table_chunks = self._chunk_tables(parsed.tables, chunk_index)
            chunks.extend(table_chunks)

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

    def _chunk_section(
        self,
        section: DocumentSection,
        start_index: int,
    ) -> list[ChunkedSection]:
        """Split a section into paragraph-aware chunks."""
        text = (section.full_content or section.content or "").strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[ChunkedSection] = []
        buffer: list[str] = []

        def flush_buffer():
            if not buffer:
                return
            content = "\n\n".join(buffer).strip()
            if not content:
                buffer.clear()
                return
            chunk = ChunkedSection(
                content=content,
                chunk_type=SECTION_TO_CHUNK_TYPE.get(
                    section.section_type, ChunkType.OTHER
                ),
                chunk_index=start_index + len(chunks),
                section_title=section.title,
                page_number=min(section.page_numbers) if section.page_numbers else None,
                chunk_metadata={"section_type": section.section_type.value},
            )
            chunks.append(chunk)
            buffer.clear()

        for paragraph in paragraphs:
            candidate = "\n\n".join(buffer + [paragraph]).strip()
            if len(candidate) <= self.max_chunk_chars:
                buffer.append(paragraph)
            else:
                flush_buffer()
                buffer.append(paragraph)

        flush_buffer()
        return chunks

    def _chunk_tables(
        self,
        tables: list[ExtractedTable],
        start_index: int,
    ) -> list[ChunkedSection]:
        """Create dedicated chunks for tables."""
        chunks: list[ChunkedSection] = []

        for idx, table in enumerate(tables):
            markdown = table.to_markdown()
            if not markdown:
                continue

            content = f"Table {table.table_id}:\n{markdown}"
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

