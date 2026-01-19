"""Chunking for medical research papers."""

from .docling_chunker import (
    DoclingHierarchicalChunker,
    get_docling_chunker,
)
from .section_chunker import (
    SECTION_TO_CHUNK_TYPE,
    ChunkedSection,
    SectionChunker,
)

__all__ = [
    # Primary chunker (Docling hierarchical)
    "DoclingHierarchicalChunker",
    "get_docling_chunker",
    # Legacy chunker
    "SectionChunker",
    # Common exports
    "ChunkedSection",
    "SECTION_TO_CHUNK_TYPE",
]


