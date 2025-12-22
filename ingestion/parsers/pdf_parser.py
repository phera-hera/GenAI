"""
PDF Parser for Medical Research Papers

Provides PDF parsing capabilities using Azure Document Intelligence
for extracting structured content from medical research PDFs,
including tables, sections, and metadata.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.exceptions import DocumentParsingError
from app.services.azure_document import AzureDocumentClient, get_document_client

from .document_result import (
    DocumentSection,
    ExtractedTable,
    PaperMetadata,
    ParsedDocument,
    SectionType,
    TableCell,
)

logger = logging.getLogger(__name__)


# Section identification patterns for medical papers
SECTION_PATTERNS: dict[SectionType, list[str]] = {
    SectionType.ABSTRACT: [
        r"^abstract$",
        r"^summary$",
        r"^synopsis$",
    ],
    SectionType.INTRODUCTION: [
        r"^introduction$",
        r"^1\.?\s*introduction$",
        r"^background$",
        r"^1\.?\s*background$",
    ],
    SectionType.BACKGROUND: [
        r"^background$",
        r"^literature\s+review$",
        r"^related\s+work$",
    ],
    SectionType.METHODS: [
        r"^methods?$",
        r"^methodology$",
        r"^materials?\s+and\s+methods?$",
        r"^2\.?\s*methods?$",
        r"^experimental\s+design$",
        r"^study\s+design$",
        r"^patients?\s+and\s+methods?$",
    ],
    SectionType.RESULTS: [
        r"^results?$",
        r"^findings$",
        r"^3\.?\s*results?$",
        r"^outcomes?$",
    ],
    SectionType.DISCUSSION: [
        r"^discussion$",
        r"^4\.?\s*discussion$",
        r"^discussion\s+and\s+conclusions?$",
    ],
    SectionType.CONCLUSION: [
        r"^conclusions?$",
        r"^concluding\s+remarks$",
        r"^5\.?\s*conclusions?$",
        r"^summary\s+and\s+conclusions?$",
    ],
    SectionType.REFERENCES: [
        r"^references?$",
        r"^bibliography$",
        r"^literature\s+cited$",
    ],
    SectionType.ACKNOWLEDGMENTS: [
        r"^acknowledgm?ents?$",
        r"^funding$",
        r"^conflicts?\s+of\s+interest$",
    ],
    SectionType.APPENDIX: [
        r"^appendix",
        r"^supplementary",
        r"^supporting\s+information$",
    ],
}


class MedicalPDFParser:
    """
    Parser for medical research PDFs using Azure Document Intelligence.
    
    This parser is optimized for medical research papers and provides:
    - Section identification (Abstract, Methods, Results, etc.)
    - Table extraction with structured data
    - Metadata extraction (title, authors, DOI)
    - Content suitable for RAG chunking
    """
    
    def __init__(
        self,
        azure_client: AzureDocumentClient | None = None,
    ):
        """
        Initialize the medical PDF parser.
        
        Args:
            azure_client: Azure Document Intelligence client (uses global if None)
        """
        self._azure_client = azure_client
    
    @property
    def azure_client(self) -> AzureDocumentClient:
        """Get the Azure Document Intelligence client."""
        if self._azure_client is None:
            self._azure_client = get_document_client()
        return self._azure_client
    
    def is_available(self) -> bool:
        """Check if the parser is available (Azure configured)."""
        return settings.is_azure_document_intelligence_configured()
    
    def parse(
        self,
        pdf_content: bytes,
        source_file: str | None = None,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse a PDF document and extract structured content.
        
        Args:
            pdf_content: PDF file content as bytes
            source_file: Optional source file path for metadata
            extract_metadata: Whether to extract paper metadata
            
        Returns:
            ParsedDocument with extracted content
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        logger.info(f"Starting PDF parsing for: {source_file or 'unknown'}")
        
        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(pdf_content).hexdigest()
        
        try:
            # Use Azure Document Intelligence for extraction
            raw_result = self.azure_client.analyze_pdf(pdf_content)
            
            # Process the raw result into structured format
            parsed = self._process_raw_result(raw_result)
            
            # Extract metadata if requested
            if extract_metadata:
                parsed.metadata = self._extract_metadata(raw_result, parsed)
            
            # Set processing metadata
            parsed.metadata.source_file = source_file
            parsed.metadata.file_hash = file_hash
            parsed.metadata.parsed_at = datetime.now(timezone.utc)
            
            logger.info(
                f"Successfully parsed PDF: {parsed.page_count} pages, "
                f"{len(parsed.sections)} sections, {len(parsed.tables)} tables"
            )
            
            return parsed
            
        except DocumentParsingError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            # Return a failed parsing result
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
    
    def parse_from_url(
        self,
        pdf_url: str,
        extract_metadata: bool = True,
    ) -> ParsedDocument:
        """
        Parse a PDF document from a URL.
        
        Args:
            pdf_url: URL to the PDF file
            extract_metadata: Whether to extract paper metadata
            
        Returns:
            ParsedDocument with extracted content
        """
        logger.info(f"Starting PDF parsing from URL: {pdf_url}")
        
        try:
            raw_result = self.azure_client.analyze_pdf_from_url(pdf_url)
            parsed = self._process_raw_result(raw_result)
            
            if extract_metadata:
                parsed.metadata = self._extract_metadata(raw_result, parsed)
            
            parsed.metadata.source_file = pdf_url
            parsed.metadata.parsed_at = datetime.now(timezone.utc)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse PDF from URL: {e}")
            return ParsedDocument(
                full_text="",
                is_successful=False,
                parsing_errors=[str(e)],
                metadata=PaperMetadata(
                    source_file=pdf_url,
                    parsed_at=datetime.now(timezone.utc),
                ),
            )
    
    def _process_raw_result(self, raw_result: dict[str, Any]) -> ParsedDocument:
        """
        Process raw Azure Document Intelligence result into ParsedDocument.
        
        Args:
            raw_result: Raw extraction result from Azure
            
        Returns:
            Structured ParsedDocument
        """
        # Extract full text
        full_text = raw_result.get("content", "")
        page_count = raw_result.get("page_count", 0)
        
        # Extract tables
        tables = self._extract_tables(raw_result.get("tables", []))
        
        # Identify and extract sections
        sections = self._extract_sections(raw_result.get("paragraphs", []))
        
        # Store raw paragraphs for debugging
        raw_paragraphs = raw_result.get("paragraphs", [])
        
        return ParsedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            page_count=page_count,
            raw_paragraphs=raw_paragraphs,
            is_successful=True,
        )
    
    def _extract_tables(
        self,
        raw_tables: list[dict[str, Any]],
    ) -> list[ExtractedTable]:
        """
        Extract and structure tables from raw data.
        
        Args:
            raw_tables: Raw table data from Azure
            
        Returns:
            List of structured ExtractedTable objects
        """
        tables = []
        
        for idx, raw_table in enumerate(raw_tables):
            cells = []
            
            for raw_cell in raw_table.get("cells", []):
                cell = TableCell(
                    content=raw_cell.get("content", ""),
                    row_index=raw_cell.get("row_index", 0),
                    column_index=raw_cell.get("column_index", 0),
                    row_span=raw_cell.get("row_span", 1),
                    column_span=raw_cell.get("column_span", 1),
                    is_header=raw_cell.get("kind") == "columnHeader",
                )
                cells.append(cell)
            
            table = ExtractedTable(
                table_id=idx + 1,
                row_count=raw_table.get("row_count", 0),
                column_count=raw_table.get("column_count", 0),
                cells=cells,
                page_number=raw_table.get("page_number"),
            )
            
            # Try to find table caption
            table.caption = self._find_table_caption(idx, raw_table)
            
            tables.append(table)
        
        logger.debug(f"Extracted {len(tables)} tables")
        return tables
    
    def _find_table_caption(
        self,
        table_idx: int,
        raw_table: dict[str, Any],
    ) -> str | None:
        """
        Attempt to find the caption for a table.
        
        Args:
            table_idx: Index of the table
            raw_table: Raw table data
            
        Returns:
            Caption text if found
        """
        # This is a simplified caption finder
        # In practice, captions might need to be found from nearby paragraphs
        return None
    
    def _extract_sections(
        self,
        paragraphs: list[dict[str, Any]],
    ) -> list[DocumentSection]:
        """
        Identify and extract sections from paragraphs.
        
        Uses pattern matching to identify section headers in
        medical papers and groups content accordingly.
        
        Args:
            paragraphs: List of paragraph data from Azure
            
        Returns:
            List of identified sections
        """
        sections: list[DocumentSection] = []
        current_section: DocumentSection | None = None
        current_content: list[str] = []
        current_pages: list[int] = []
        
        for para in paragraphs:
            content = para.get("content", "").strip()
            role = para.get("role", "")
            page_num = para.get("page_number")
            
            if not content:
                continue
            
            # Check if this is a section header
            section_type = self._identify_section_type(content, role)
            
            if section_type:
                # Save previous section if exists
                if current_section is not None:
                    current_section.content = "\n\n".join(current_content)
                    current_section.page_numbers = list(set(current_pages))
                    sections.append(current_section)
                
                # Start new section
                current_section = DocumentSection(
                    section_type=section_type,
                    title=content,
                    content="",
                    paragraphs=[],
                )
                current_content = []
                current_pages = []
            else:
                # Add content to current section
                current_content.append(content)
                if page_num:
                    current_pages.append(page_num)
        
        # Don't forget the last section
        if current_section is not None:
            current_section.content = "\n\n".join(current_content)
            current_section.page_numbers = list(set(current_pages))
            sections.append(current_section)
        elif current_content:
            # Content without identified sections goes to "other"
            sections.append(
                DocumentSection(
                    section_type=SectionType.OTHER,
                    title=None,
                    content="\n\n".join(current_content),
                    page_numbers=list(set(current_pages)),
                )
            )
        
        logger.debug(f"Identified {len(sections)} sections")
        return sections
    
    def _identify_section_type(
        self,
        text: str,
        role: str,
    ) -> SectionType | None:
        """
        Identify the section type from header text.
        
        Args:
            text: Header text
            role: Paragraph role from Azure (e.g., "sectionHeading")
            
        Returns:
            SectionType if identified, None otherwise
        """
        # Only consider likely headers
        if role not in ["sectionHeading", "title", ""] or len(text) > 100:
            return None
        
        # Check for numbered sections (1., 2., I., II., etc.)
        text_clean = re.sub(r"^(\d+\.?|\[?\d+\]?|[IVX]+\.?)\s*", "", text.strip())
        text_lower = text_clean.lower().strip()
        
        # Match against known section patterns
        for section_type, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, text_lower, re.IGNORECASE):
                    return section_type
        
        # Additional heuristics for section headers
        if self._looks_like_header(text):
            # Check for specific keywords that might indicate a section
            if any(kw in text_lower for kw in ["method", "material", "procedure"]):
                return SectionType.METHODS
            if any(kw in text_lower for kw in ["result", "finding", "outcome"]):
                return SectionType.RESULTS
            if "discuss" in text_lower:
                return SectionType.DISCUSSION
            if any(kw in text_lower for kw in ["conclus", "summary"]):
                return SectionType.CONCLUSION
        
        return None
    
    def _looks_like_header(self, text: str) -> bool:
        """
        Check if text appears to be a section header.
        
        Args:
            text: Text to check
            
        Returns:
            True if text looks like a header
        """
        text = text.strip()
        
        # Short text is more likely a header
        if len(text) > 80:
            return False
        
        # Numbered headings (1., 2., 1.1, etc.)
        if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text):
            return True
        
        # Roman numeral headings
        if re.match(r"^[IVX]+\.?\s+[A-Z]", text):
            return True
        
        # All caps text (common for headers)
        if text.isupper() and len(text) > 3:
            return True
        
        # Title case without period at end
        words = text.split()
        if len(words) <= 6 and not text.endswith("."):
            if words and words[0][0].isupper():
                return True
        
        return False
    
    def _extract_metadata(
        self,
        raw_result: dict[str, Any],
        parsed: ParsedDocument,
    ) -> PaperMetadata:
        """
        Extract paper metadata from the parsed content.
        
        Attempts to identify title, authors, abstract, and other
        bibliographic information from the parsed content.
        
        Args:
            raw_result: Raw Azure extraction result
            parsed: Already processed ParsedDocument
            
        Returns:
            Extracted PaperMetadata
        """
        metadata = PaperMetadata()
        
        paragraphs = raw_result.get("paragraphs", [])
        
        # Extract title (usually first title-role paragraph)
        for para in paragraphs:
            role = para.get("role", "")
            content = para.get("content", "").strip()
            
            if role == "title" and content:
                metadata.title = content
                break
        
        # Fallback: Use first substantial text as title
        if not metadata.title:
            for para in paragraphs[:5]:  # Check first 5 paragraphs
                content = para.get("content", "").strip()
                if content and 10 < len(content) < 300:
                    metadata.title = content
                    break
        
        # Extract abstract from sections
        abstract_section = parsed.get_section(SectionType.ABSTRACT)
        if abstract_section:
            metadata.abstract = abstract_section.content
        
        # Extract DOI if present in text
        doi_match = re.search(
            r"(?:doi[:\s]*)?10\.\d{4,}/[^\s]+",
            parsed.full_text,
            re.IGNORECASE,
        )
        if doi_match:
            doi = doi_match.group().strip()
            # Clean up DOI
            doi = re.sub(r"^doi[:\s]*", "", doi, flags=re.IGNORECASE)
            metadata.doi = doi
        
        # Extract keywords if present
        keywords_match = re.search(
            r"keywords?[:\s]*([^\n]+)",
            parsed.full_text,
            re.IGNORECASE,
        )
        if keywords_match:
            keywords_text = keywords_match.group(1)
            # Split by common delimiters
            keywords = re.split(r"[,;•]", keywords_text)
            metadata.keywords = [kw.strip() for kw in keywords if kw.strip()]
        
        # Try to extract authors from key-value pairs
        for kv in raw_result.get("key_value_pairs", []):
            key = kv.get("key", "").lower()
            value = kv.get("value", "")
            
            if "author" in key and value:
                # Split authors by common patterns
                authors = re.split(r"[,;]|\band\b", value)
                metadata.authors = [a.strip() for a in authors if a.strip()]
        
        return metadata


# Global parser instance
_pdf_parser: MedicalPDFParser | None = None


def get_pdf_parser() -> MedicalPDFParser:
    """Get or create the global PDF parser instance."""
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = MedicalPDFParser()
    return _pdf_parser

