"""
Fallback PDF Parser using PyMuPDF

Provides a lightweight fallback PDF parser for when LlamaParser
is unavailable. Uses PyMuPDF (fitz) for basic text
and table extraction.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

from .document_result import (
    DocumentSection,
    ExtractedTable,
    PaperMetadata,
    ParsedDocument,
    SectionType,
    TableCell,
)

logger = logging.getLogger(__name__)


# Section identification patterns (same as main parser)
SECTION_PATTERNS: dict[SectionType, list[str]] = {
    SectionType.ABSTRACT: [
        r"^abstract$",
        r"^summary$",
    ],
    SectionType.INTRODUCTION: [
        r"^introduction$",
        r"^1\.?\s*introduction$",
        r"^background$",
    ],
    SectionType.METHODS: [
        r"^methods?$",
        r"^methodology$",
        r"^materials?\s+and\s+methods?$",
        r"^2\.?\s*methods?$",
    ],
    SectionType.RESULTS: [
        r"^results?$",
        r"^findings$",
        r"^3\.?\s*results?$",
    ],
    SectionType.DISCUSSION: [
        r"^discussion$",
        r"^4\.?\s*discussion$",
    ],
    SectionType.CONCLUSION: [
        r"^conclusions?$",
        r"^5\.?\s*conclusions?$",
    ],
    SectionType.REFERENCES: [
        r"^references?$",
        r"^bibliography$",
    ],
}


class FallbackPDFParser:
    """
    Fallback PDF parser using PyMuPDF.
    
    This parser provides basic PDF extraction capabilities when
    LlamaParser is not available. It's less
    accurate but works offline and has no API costs.
    
    Note: Requires pymupdf package to be installed.
    """
    
    def __init__(self):
        """Initialize the fallback parser."""
        self._fitz_available: bool | None = None
    
    def is_available(self) -> bool:
        """Check if PyMuPDF is available."""
        if self._fitz_available is None:
            try:
                import fitz  # noqa: F401
                self._fitz_available = True
            except ImportError:
                self._fitz_available = False
                logger.warning(
                    "PyMuPDF not available. Install with: pip install pymupdf"
                )
        return self._fitz_available
    
    def parse(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse a PDF document using PyMuPDF.
        
        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path
            extract_metadata: Whether to extract metadata
            
        Returns:
            ParsedDocument with extracted content
        """
        if not self.is_available():
            return ParsedDocument(
                full_text="",
                is_successful=False,
                parsing_errors=["PyMuPDF is not installed"],
                metadata=PaperMetadata(
                    source_file=source_file,
                    parsed_at=datetime.now(timezone.utc),
                ),
            )
        
        import fitz
        
        logger.info(f"Fallback parsing PDF: {source_file or 'unknown'}")
        
        # Calculate file hash
        file_hash = hashlib.sha256(pdf_content).hexdigest()
        
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Extract text from all pages
            full_text_parts = []
            page_texts: list[dict[str, Any]] = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                full_text_parts.append(text)
                page_texts.append({
                    "page_number": page_num + 1,
                    "text": text,
                })
            
            full_text = "\n\n".join(full_text_parts)
            
            # Extract tables (basic extraction)
            tables = self._extract_tables(doc)
            
            # Identify sections from text
            sections = self._extract_sections(full_text)
            
            # Create metadata
            metadata = PaperMetadata(
                source_file=source_file,
                file_hash=file_hash,
                parsed_at=datetime.now(timezone.utc),
                parser_version="1.0.0-fallback",
            )
            
            # Try to extract PDF metadata
            if extract_metadata:
                pdf_metadata = doc.metadata
                if pdf_metadata:
                    metadata.title = pdf_metadata.get("title") or None
                    author = pdf_metadata.get("author")
                    if author:
                        metadata.authors = [a.strip() for a in author.split(",")]
            
            # Try to extract title from first page
            if not metadata.title and page_texts:
                first_page = page_texts[0]["text"]
                lines = [l.strip() for l in first_page.split("\n") if l.strip()]
                if lines:
                    # First substantial line is likely the title
                    for line in lines[:5]:
                        if 10 < len(line) < 200:
                            metadata.title = line
                            break
            
            # Extract abstract
            abstract_section = None
            for section in sections:
                if section.section_type == SectionType.ABSTRACT:
                    abstract_section = section
                    break
            
            if abstract_section:
                metadata.abstract = abstract_section.content
            
            doc.close()
            
            logger.info(
                f"Fallback parsed PDF: {len(page_texts)} pages, "
                f"{len(sections)} sections, {len(tables)} tables"
            )
            
            return ParsedDocument(
                full_text=full_text,
                sections=sections,
                tables=tables,
                metadata=metadata,
                page_count=len(page_texts),
                is_successful=True,
                parsing_warnings=["Parsed with fallback parser - reduced accuracy"],
            )
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return ParsedDocument(
                full_text="",
                is_successful=False,
                parsing_errors=[str(e)],
                metadata=PaperMetadata(
                    source_file=source_file,
                    file_hash=file_hash,
                    parsed_at=datetime.now(timezone.utc),
                ),
            )
    
    def _extract_tables(self, doc: Any) -> list[ExtractedTable]:
        """
        Extract tables from PDF using PyMuPDF.
        
        Note: PyMuPDF's table extraction is basic compared to
        LlamaParser. Complex tables may not
        be properly extracted.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of extracted tables
        """
        tables: list[ExtractedTable] = []
        table_id = 0
        
        try:
            for page_num, page in enumerate(doc):
                # Try to find tables on the page
                # This uses PyMuPDF's table detection which is basic
                page_tables = page.find_tables()
                
                for tab in page_tables:
                    table_id += 1
                    cells = []
                    
                    # Extract table data
                    table_data = tab.extract()
                    
                    if not table_data:
                        continue
                    
                    row_count = len(table_data)
                    col_count = max(len(row) for row in table_data) if table_data else 0
                    
                    for row_idx, row in enumerate(table_data):
                        for col_idx, cell_content in enumerate(row):
                            cell = TableCell(
                                content=str(cell_content) if cell_content else "",
                                row_index=row_idx,
                                column_index=col_idx,
                                is_header=(row_idx == 0),  # Assume first row is header
                            )
                            cells.append(cell)
                    
                    if cells:
                        tables.append(
                            ExtractedTable(
                                table_id=table_id,
                                row_count=row_count,
                                column_count=col_count,
                                cells=cells,
                                page_number=page_num + 1,
                            )
                        )
        
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    def _extract_sections(self, full_text: str) -> list[DocumentSection]:
        """
        Extract sections from the full text.
        
        Uses pattern matching to identify section headers
        and group content.
        
        Args:
            full_text: Complete document text
            
        Returns:
            List of identified sections
        """
        sections: list[DocumentSection] = []
        lines = full_text.split("\n")
        
        current_section_type: SectionType | None = None
        current_section_title: str | None = None
        current_content: list[str] = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            section_type = self._identify_section(line)
            
            if section_type:
                # Save previous section
                if current_content:
                    sections.append(
                        DocumentSection(
                            section_type=current_section_type or SectionType.OTHER,
                            title=current_section_title,
                            content="\n".join(current_content),
                        )
                    )
                
                # Start new section
                current_section_type = section_type
                current_section_title = line
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget last section
        if current_content:
            sections.append(
                DocumentSection(
                    section_type=current_section_type or SectionType.OTHER,
                    title=current_section_title,
                    content="\n".join(current_content),
                )
            )
        
        return sections
    
    def _identify_section(self, text: str) -> SectionType | None:
        """
        Identify if text is a section header.
        
        Args:
            text: Line of text to check
            
        Returns:
            SectionType if identified as header
        """
        # Skip long lines
        if len(text) > 80:
            return None
        
        # Clean up text for matching
        text_clean = re.sub(r"^(\d+\.?|\[?\d+\]?|[IVX]+\.?)\s*", "", text)
        text_lower = text_clean.lower().strip()
        
        # Check against patterns
        for section_type, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, text_lower, re.IGNORECASE):
                    return section_type
        
        return None


# Global fallback parser instance
_fallback_parser: FallbackPDFParser | None = None


def get_fallback_parser() -> FallbackPDFParser:
    """Get or create the global fallback parser instance."""
    global _fallback_parser
    if _fallback_parser is None:
        _fallback_parser = FallbackPDFParser()
    return _fallback_parser

