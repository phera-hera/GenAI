"""
PDF Parser Facade

Provides a unified interface for PDF parsing that automatically
selects the best available parser based on configuration.

Priority:
1. Docling (primary - vision-based with hierarchical structure)
2. PyMuPDF fallback (offline, basic extraction)
3. LlamaParser (deprecated, fallback)
"""

import logging
from typing import Protocol

from medical_agent.core.config import settings

from .document_result import ParsedDocument
from .docling_parser import DoclingPDFParser, get_docling_parser
from .fallback_parser import FallbackPDFParser, get_fallback_parser
from .pdf_parser import MedicalPDFParser, get_pdf_parser

logger = logging.getLogger(__name__)


class PDFParserProtocol(Protocol):
    """Protocol defining the PDF parser interface."""

    def is_available(self) -> bool:
        """Check if the parser is available."""
        ...

    def parse(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """Parse PDF content and return structured result."""
        ...


class PDFParserFacade:
    """
    Unified PDF parser that selects the best available parser.

    Automatically chooses between Docling (primary), PyMuPDF (fallback),
    and LlamaParser (deprecated) based on availability and configuration.

    Usage:
        parser = PDFParserFacade()
        result = parser.parse(pdf_bytes, source_file="paper.pdf")
    """

    def __init__(
        self,
        prefer_docling: bool = True,
        allow_fallback: bool = True,
    ):
        """
        Initialize the parser facade.

        Args:
            prefer_docling: Whether to prefer Docling over other parsers
            allow_fallback: Whether to allow fallback parser
        """
        self.prefer_docling = prefer_docling
        self.allow_fallback = allow_fallback

        self._docling_parser: DoclingPDFParser | None = None
        self._llama_parser: MedicalPDFParser | None = None
        self._fallback_parser: FallbackPDFParser | None = None

    @property
    def docling_parser(self) -> DoclingPDFParser:
        """Get the Docling parser instance."""
        if self._docling_parser is None:
            self._docling_parser = get_docling_parser()
        return self._docling_parser

    @property
    def llama_parser(self) -> MedicalPDFParser:
        """Get the LlamaParser instance (deprecated)."""
        if self._llama_parser is None:
            self._llama_parser = get_pdf_parser()
        return self._llama_parser

    @property
    def fallback_parser(self) -> FallbackPDFParser:
        """Get the fallback parser instance."""
        if self._fallback_parser is None:
            self._fallback_parser = get_fallback_parser()
        return self._fallback_parser

    def get_available_parser(self) -> PDFParserProtocol | None:
        """
        Get the best available parser.

        Returns:
            The best available parser, or None if none available
        """
        # Check Docling first if preferred
        if self.prefer_docling and self.docling_parser.is_available():
            return self.docling_parser

        # Try PyMuPDF fallback
        if self.allow_fallback and self.fallback_parser.is_available():
            return self.fallback_parser

        # Try LlamaParser as last resort (deprecated)
        if self.llama_parser.is_available():
            return self.llama_parser

        return None

    def is_available(self) -> bool:
        """Check if any parser is available."""
        return self.get_available_parser() is not None

    def get_parser_name(self) -> str:
        """Get the name of the parser that will be used."""
        parser = self.get_available_parser()
        if parser is None:
            return "none"
        if isinstance(parser, DoclingPDFParser):
            return "docling"
        if isinstance(parser, MedicalPDFParser):
            return "llama_parser"
        if isinstance(parser, FallbackPDFParser):
            return "fallback"
        return "unknown"
    
    def parse(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse a PDF document using the best available parser.

        Automatically falls back to PyMuPDF if Docling fails.

        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path for metadata
            extract_metadata: Whether to extract paper metadata

        Returns:
            ParsedDocument with extracted content
        """
        from datetime import datetime, timezone

        from .document_result import PaperMetadata

        # Try Docling first if available
        if self.docling_parser.is_available():
            parser_name = "docling"
            logger.info(f"Using {parser_name} parser for: {source_file or 'unknown'}")

            try:
                result = self.docling_parser.parse(
                    pdf_content=pdf_content,
                    source_file=source_file,
                    extract_metadata=extract_metadata,
                )

                # If parsing was successful, return it
                if result.is_successful and result.full_text:
                    return result

                # If failed but no fallback allowed, return the failed result
                if not self.allow_fallback:
                    logger.warning(f"Docling parsing failed and no fallback allowed")
                    return result

                # Otherwise fall through to try fallback
                logger.warning(f"Docling parsing failed, attempting fallback parser")

            except Exception as e:
                logger.warning(f"Docling failed with error: {e}")
                if not self.allow_fallback:
                    raise

        # Try fallback parser
        if self.allow_fallback and self.fallback_parser.is_available():
            parser_name = "fallback"
            logger.info(f"Using {parser_name} parser for: {source_file or 'unknown'}")

            result = self.fallback_parser.parse(
                pdf_content=pdf_content,
                source_file=source_file,
                extract_metadata=extract_metadata,
            )

            # Add fallback marker to parser version if successful
            if result.is_successful and "fallback" not in result.metadata.parser_version:
                result.metadata.parser_version += "-fallback"

            return result

        # No parser available
        logger.error("No PDF parser available")
        return ParsedDocument(
            full_text="",
            is_successful=False,
            parsing_errors=[
                "No PDF parser available. Install Docling (pip install docling) "
                "or PyMuPDF (pip install pymupdf) for fallback."
            ],
            metadata=PaperMetadata(
                source_file=source_file,
                parsed_at=datetime.now(timezone.utc),
            ),
        )
    
    def parse_with_docling(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse using Docling specifically.

        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path
            extract_metadata: Whether to extract metadata

        Returns:
            ParsedDocument with extracted content

        Raises:
            RuntimeError: If Docling is not available
        """
        if not self.docling_parser.is_available():
            raise RuntimeError(
                "Docling is not available. "
                "Install with: pip install docling docling-core"
            )

        return self.docling_parser.parse(
            pdf_content=pdf_content,
            source_file=source_file,
            extract_metadata=extract_metadata,
        )

    def parse_with_llama_parser(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse using LlamaParser specifically (deprecated).

        DEPRECATED: Use parse_with_docling() instead.

        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path
            extract_metadata: Whether to extract metadata

        Returns:
            ParsedDocument with extracted content

        Raises:
            RuntimeError: If LlamaParser is not available
        """
        logger.warning("parse_with_llama_parser is deprecated. Use parse_with_docling instead.")
        if not self.llama_parser.is_available():
            raise RuntimeError(
                "LlamaParser is not configured. "
                "Set LLAMA_CLOUD_API_KEY environment variable."
            )

        return self.llama_parser.parse(
            pdf_content=pdf_content,
            source_file=source_file,
            extract_metadata=extract_metadata,
        )
    
    def parse_with_fallback(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse using the fallback parser specifically.
        
        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path
            extract_metadata: Whether to extract metadata
            
        Returns:
            ParsedDocument with extracted content
            
        Raises:
            RuntimeError: If fallback parser is not available
        """
        if not self.fallback_parser.is_available():
            raise RuntimeError(
                "Fallback parser is not available. "
                "Install PyMuPDF: pip install pymupdf"
            )
        
        return self.fallback_parser.parse(
            pdf_content=pdf_content,
            source_file=source_file,
            extract_metadata=extract_metadata,
        )


# Global facade instance
_parser_facade: PDFParserFacade | None = None


def get_parser_facade() -> PDFParserFacade:
    """Get or create the global parser facade instance."""
    global _parser_facade
    if _parser_facade is None:
        _parser_facade = PDFParserFacade()
    return _parser_facade


def parse_pdf(
    pdf_content: bytes,
    source_file: str | None = None,
    extract_metadata: bool = True,
) -> ParsedDocument:
    """
    Convenience function to parse a PDF document.
    
    Uses the global parser facade to select the best
    available parser automatically.
    
    Args:
        pdf_content: PDF file content as bytes
        source_file: Optional source file path
        extract_metadata: Whether to extract metadata
        
    Returns:
        ParsedDocument with extracted content
    """
    facade = get_parser_facade()
    return facade.parse(
        pdf_content=pdf_content,
        source_file=source_file,
        extract_metadata=extract_metadata,
    )
