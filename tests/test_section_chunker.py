"""
Tests for section-based chunking of parsed medical papers.
"""

from app.db.models import ChunkType
from ingestion.chunkers import ChunkedSection, SectionChunker
from ingestion.parsers import (
    DocumentSection,
    ExtractedTable,
    PaperMetadata,
    ParsedDocument,
    SectionType,
    TableCell,
)


def _example_parsed_document() -> ParsedDocument:
    """Build a small parsed document for chunking tests."""
    abstract = DocumentSection(
        section_type=SectionType.ABSTRACT,
        title="Abstract",
        content="Short abstract text.",
    )

    intro_text = (
        "Intro paragraph one with enough words to exceed the chunk size limit."
        "\n\n"
        "Intro paragraph two continues the introduction and should be split."
    )
    introduction = DocumentSection(
        section_type=SectionType.INTRODUCTION,
        title="Introduction",
        content=intro_text,
        page_numbers=[1],
    )

    methods = DocumentSection(
        section_type=SectionType.METHODS,
        title="Methods",
        content="Methods content goes here.",
        page_numbers=[3, 4],
    )

    table = ExtractedTable(
        table_id=1,
        row_count=2,
        column_count=2,
        cells=[
            TableCell("Header A", 0, 0, is_header=True),
            TableCell("Header B", 0, 1, is_header=True),
            TableCell("Value 1", 1, 0),
            TableCell("Value 2", 1, 1),
        ],
        page_number=2,
        caption="Study metrics",
    )

    metadata = PaperMetadata(
        title="Test Paper",
        authors=["Author One", "Author Two"],
        doi="10.1234/example",
        publication_year=2024,
        journal="Medical Journal",
        abstract="Metadata abstract text.",
    )

    return ParsedDocument(
        full_text="Full text placeholder",
        sections=[abstract, introduction, methods],
        tables=[table],
        metadata=metadata,
        page_count=4,
    )


class TestSectionChunker:
    """Unit tests for the SectionChunker."""

    def test_chunking_produces_expected_structure(self):
        """Chunker should emit metadata, section, and table chunks in order."""
        parsed = _example_parsed_document()
        chunker = SectionChunker(max_chunk_chars=80, include_tables=True)

        chunks = chunker.chunk_document(parsed)

        # One metadata/abstract chunk + intro split into 2 + methods + table
        assert len(chunks) == 5
        assert chunks[0].chunk_type == ChunkType.ABSTRACT
        assert "Title: Test Paper" in chunks[0].content
        assert "Abstract:" in chunks[0].content

        intro_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTRODUCTION]
        assert len(intro_chunks) == 2  # split across paragraphs/length
        assert intro_chunks[0].section_title == "Introduction"
        assert intro_chunks[0].page_number == 1

        methods_chunk = next(c for c in chunks if c.chunk_type == ChunkType.METHODS)
        assert methods_chunk.page_number == 3
        assert "Methods content" in methods_chunk.content

        table_chunk = next(c for c in chunks if c.chunk_type == ChunkType.TABLE)
        assert "Table 1" in table_chunk.content
        assert table_chunk.chunk_metadata["table_id"] == 1
        assert table_chunk.section_title == "Study metrics"

    def test_chunk_metadata_fields_present(self):
        """Chunk metadata should carry section type and ordering."""
        parsed = _example_parsed_document()
        chunker = SectionChunker(max_chunk_chars=300)

        chunks = chunker.chunk_document(parsed)

        # Ensure chunk indices are sequential
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

        for chunk in chunks:
            assert isinstance(chunk, ChunkedSection)
            assert chunk.content.strip() != ""
            assert isinstance(chunk.chunk_metadata, dict)
            if chunk.chunk_type == ChunkType.METHODS:
                assert chunk.chunk_metadata["section_type"] == SectionType.METHODS.value

