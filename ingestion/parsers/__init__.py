"""
Document Parsers for Medical Research Papers

Provides PDF parsing capabilities using LlamaParser (LlamaCloud)
as the primary parser, with a PyMuPDF fallback for offline use.

Main components:
- MedicalPDFParser: Primary parser using LlamaParser
- FallbackPDFParser: Backup parser using PyMuPDF (offline capable)
- PDFParserFacade: Unified interface that auto-selects best parser
- ParsedDocument: Structured representation of parsed papers
- ExtractedTable: Table data with markdown export
- DocumentSection: Identified paper sections

Quick start:
    from ingestion.parsers import parse_pdf
    
    result = parse_pdf(pdf_bytes, source_file="paper.pdf")
    print(result.summary())
"""

from .document_result import (
    DocumentSection,
    ExtractedTable,
    PaperMetadata,
    ParsedDocument,
    SectionType,
    TableCell,
)
from .fallback_parser import FallbackPDFParser, get_fallback_parser
from .parser_facade import PDFParserFacade, get_parser_facade, parse_pdf
from .pdf_parser import MedicalPDFParser, get_pdf_parser

__all__ = [
    # Convenience function
    "parse_pdf",
    # Parser facade
    "PDFParserFacade",
    "get_parser_facade",
    # Main parser
    "MedicalPDFParser",
    "get_pdf_parser",
    # Fallback parser
    "FallbackPDFParser",
    "get_fallback_parser",
    # Data classes
    "ParsedDocument",
    "DocumentSection",
    "ExtractedTable",
    "TableCell",
    "PaperMetadata",
    "SectionType",
]
