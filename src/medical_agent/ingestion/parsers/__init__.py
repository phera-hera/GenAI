"""Type definitions for document parsing - used by metadata utilities."""

from .document_result import (
    DocumentSection,
    ExtractedTable,
    PaperMetadata,
    ParsedDocument,
    SectionType,
    TableCell,
)

__all__ = [
    "DocumentSection",
    "ExtractedTable",
    "PaperMetadata",
    "ParsedDocument",
    "SectionType",
    "TableCell",
]
