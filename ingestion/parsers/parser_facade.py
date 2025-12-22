"""
PDF Parser Facade

Provides a unified interface for PDF parsing that automatically
selects the best available parser based on configuration.

Priority:
1. Azure Document Intelligence (primary - better quality)
2. PyMuPDF fallback (offline, basic extraction)
"""

import logging
from typing import Protocol

from app.core.config import settings

from .document_result import ParsedDocument
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
    
    Automatically chooses between Azure Document Intelligence
    and PyMuPDF based on availability and configuration.
    
    Usage:
        parser = PDFParserFacade()
        result = parser.parse(pdf_bytes, source_file="paper.pdf")
    """
    
    def __init__(
        self,
        prefer_azure: bool = True,
        allow_fallback: bool = True,
    ):
        """
        Initialize the parser facade.
        
        Args:
            prefer_azure: Whether to prefer Azure over fallback
            allow_fallback: Whether to allow fallback parser
        """
        self.prefer_azure = prefer_azure
        self.allow_fallback = allow_fallback
        
        self._azure_parser: MedicalPDFParser | None = None
        self._fallback_parser: FallbackPDFParser | None = None
    
    @property
    def azure_parser(self) -> MedicalPDFParser:
        """Get the Azure parser instance."""
        if self._azure_parser is None:
            self._azure_parser = get_pdf_parser()
        return self._azure_parser
    
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
        # Check Azure first if preferred
        if self.prefer_azure and self.azure_parser.is_available():
            return self.azure_parser
        
        # Try fallback
        if self.allow_fallback and self.fallback_parser.is_available():
            return self.fallback_parser
        
        # If not preferring Azure, try it as last resort
        if not self.prefer_azure and self.azure_parser.is_available():
            return self.azure_parser
        
        return None
    
    def is_available(self) -> bool:
        """Check if any parser is available."""
        return self.get_available_parser() is not None
    
    def get_parser_name(self) -> str:
        """Get the name of the parser that will be used."""
        parser = self.get_available_parser()
        if parser is None:
            return "none"
        if isinstance(parser, MedicalPDFParser):
            return "azure"
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
        
        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path for metadata
            extract_metadata: Whether to extract paper metadata
            
        Returns:
            ParsedDocument with extracted content
        """
        parser = self.get_available_parser()
        
        if parser is None:
            logger.error("No PDF parser available")
            from .document_result import PaperMetadata
            from datetime import datetime, timezone
            
            return ParsedDocument(
                full_text="",
                is_successful=False,
                parsing_errors=[
                    "No PDF parser available. Configure Azure Document Intelligence "
                    "or install PyMuPDF (pip install pymupdf) for fallback."
                ],
                metadata=PaperMetadata(
                    source_file=source_file,
                    parsed_at=datetime.now(timezone.utc),
                ),
            )
        
        parser_name = self.get_parser_name()
        logger.info(f"Using {parser_name} parser for: {source_file or 'unknown'}")
        
        result = parser.parse(
            pdf_content=pdf_content,
            source_file=source_file,
            extract_metadata=extract_metadata,
        )
        
        # Add parser info to metadata
        if parser_name == "fallback" and result.is_successful:
            if "fallback" not in result.metadata.parser_version:
                result.metadata.parser_version += "-fallback"
        
        return result
    
    def parse_with_azure(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse using Azure Document Intelligence specifically.
        
        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path
            extract_metadata: Whether to extract metadata
            
        Returns:
            ParsedDocument with extracted content
            
        Raises:
            RuntimeError: If Azure parser is not available
        """
        if not self.azure_parser.is_available():
            raise RuntimeError(
                "Azure Document Intelligence is not configured. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables."
            )
        
        return self.azure_parser.parse(
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

