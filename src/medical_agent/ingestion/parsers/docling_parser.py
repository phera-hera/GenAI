"""
Docling PDF Parser for Medical Research Papers

Provides PDF parsing capabilities using Docling with hybrid OCR
for extracting structured content from medical research PDFs,
including tables, sections, and metadata with hierarchical structure.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from medical_agent.core.config import settings
from medical_agent.core.exceptions import DocumentParsingError

from .document_result import (
    DocumentSection,
    ExtractedTable,
    PaperMetadata,
    ParsedDocument,
    SectionType,
    TableCell,
)

logger = logging.getLogger(__name__)


class DoclingPDFParser:
    """
    Parser for medical research PDFs using Docling.

    This parser uses Docling's vision-based parsing with hybrid OCR for:
    - Hierarchical section identification
    - Table extraction with TableFormer
    - Metadata extraction (title, authors, DOI, keywords, abstract)
    - Content suitable for hierarchical chunking
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        use_tableformer: bool = True,
    ):
        """
        Initialize the Docling PDF parser.

        Args:
            enable_ocr: Enable hybrid OCR for scanned papers
            use_tableformer: Use TableFormer for table extraction
        """
        self.enable_ocr = enable_ocr
        self.use_tableformer = use_tableformer
        self._converter: DocumentConverter | None = None

    @property
    def converter(self) -> DocumentConverter:
        """Get or create the Docling document converter."""
        if self._converter is None:
            # Configure pipeline options
            pipeline_options = PdfPipelineOptions()

            # Enable OCR if requested
            pipeline_options.do_ocr = self.enable_ocr

            # Configure table extraction
            if self.use_tableformer:
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

            # Create converter with PDF format options
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )

        return self._converter

    def is_available(self) -> bool:
        """Check if the parser is available (Docling configured)."""
        try:
            # Test if we can create a converter
            _ = self.converter
            return True
        except Exception as e:
            logger.warning(f"Docling parser not available: {e}")
            return False

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
        logger.info(f"Starting Docling PDF parsing for: {source_file or 'unknown'}")

        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(pdf_content).hexdigest()

        try:
            # Save bytes to temporary file for Docling
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = tmp_file.name

            try:
                # Use Docling to parse the PDF
                result = self.converter.convert(tmp_path)

                # Process the raw result into structured format
                parsed = self._process_docling_result(result)

                # Extract metadata if requested
                if extract_metadata:
                    parsed.metadata = self._extract_metadata(result, parsed)

                # Set processing metadata
                parsed.metadata.source_file = source_file
                parsed.metadata.file_hash = file_hash
                parsed.metadata.parsed_at = datetime.now(timezone.utc)
                parsed.metadata.parser_version = "docling-1.0.0"

                logger.info(
                    f"Successfully parsed PDF: {parsed.page_count} pages, "
                    f"{len(parsed.sections)} sections, {len(parsed.tables)} tables"
                )

                return parsed

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except DocumentParsingError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF with Docling: {e}")
            # Return a failed parsing result
            return ParsedDocument(
                full_text="",
                is_successful=False,
                parsing_errors=[str(e)],
                metadata=PaperMetadata(
                    source_file=source_file,
                    file_hash=file_hash,
                    parsed_at=datetime.now(timezone.utc),
                    parser_version="docling-1.0.0",
                ),
            )

    def _process_docling_result(self, result: Any) -> ParsedDocument:
        """
        Process raw Docling result into ParsedDocument.

        Args:
            result: Raw extraction result from Docling

        Returns:
            Structured ParsedDocument
        """
        # Get the document
        doc = result.document

        # Extract full text
        full_text = doc.export_to_markdown()

        # Extract sections using Docling's hierarchical structure
        sections = self._extract_sections(doc)

        # Extract tables
        tables = self._extract_tables(doc)

        # Get page count
        page_count = len(doc.pages) if hasattr(doc, 'pages') else 0

        return ParsedDocument(
            full_text=full_text,
            sections=sections,
            tables=tables,
            page_count=page_count,
            is_successful=True,
        )

    def _extract_sections(self, doc: Any) -> list[DocumentSection]:
        """
        Extract sections from Docling document using hierarchical structure.

        Args:
            doc: Docling document object

        Returns:
            List of identified sections
        """
        sections: list[DocumentSection] = []
        current_section: DocumentSection | None = None
        current_content: list[str] = []

        # Iterate through document items
        for item, level in doc.iterate_items():
            # Check if item is a heading
            if item.label == "section_header" or (hasattr(item, 'self_ref') and 'heading' in str(item.self_ref).lower()):
                # Save previous section if exists
                if current_section is not None:
                    current_section.content = "\n\n".join(current_content)
                    sections.append(current_section)

                # Get section text
                section_text = item.text if hasattr(item, 'text') else str(item)

                # Identify section type
                section_type = self._identify_section_type(section_text)

                # Start new section
                current_section = DocumentSection(
                    section_type=section_type,
                    title=section_text,
                    content="",
                    paragraphs=[],
                )
                current_content = []

            elif item.label in ["text", "paragraph", "list_item"]:
                # Add content to current section
                text = item.text if hasattr(item, 'text') else str(item)
                if text.strip():
                    current_content.append(text.strip())

        # Don't forget the last section
        if current_section is not None:
            current_section.content = "\n\n".join(current_content)
            sections.append(current_section)
        elif current_content:
            # Content without identified sections goes to "other"
            sections.append(
                DocumentSection(
                    section_type=SectionType.OTHER,
                    title=None,
                    content="\n\n".join(current_content),
                )
            )

        logger.debug(f"Identified {len(sections)} sections using Docling structure")
        return sections

    def _identify_section_type(self, text: str) -> SectionType:
        """
        Identify the section type from header text.

        Args:
            text: Header text

        Returns:
            SectionType
        """
        import re

        # Clean up numbering
        text_clean = re.sub(r"^(\d+\.?|\[?\d+\]?|[IVX]+\.?)\s*", "", text.strip())
        text_lower = text_clean.lower().strip()

        # Match against known patterns
        if re.match(r"^abstract$", text_lower, re.IGNORECASE):
            return SectionType.ABSTRACT
        elif re.match(r"^(introduction|background)$", text_lower, re.IGNORECASE):
            return SectionType.INTRODUCTION
        elif re.match(r"^(methods?|methodology|materials?\s+and\s+methods?)$", text_lower, re.IGNORECASE):
            return SectionType.METHODS
        elif re.match(r"^(results?|findings)$", text_lower, re.IGNORECASE):
            return SectionType.RESULTS
        elif re.match(r"^discussion$", text_lower, re.IGNORECASE):
            return SectionType.DISCUSSION
        elif re.match(r"^(conclusions?|summary)$", text_lower, re.IGNORECASE):
            return SectionType.CONCLUSION
        elif re.match(r"^(references?|bibliography)$", text_lower, re.IGNORECASE):
            return SectionType.REFERENCES
        elif re.match(r"^(acknowledgm?ents?|funding)$", text_lower, re.IGNORECASE):
            return SectionType.ACKNOWLEDGMENTS
        elif re.match(r"^(appendix|supplementary)", text_lower, re.IGNORECASE):
            return SectionType.APPENDIX
        else:
            return SectionType.OTHER

    def _extract_tables(self, doc: Any) -> list[ExtractedTable]:
        """
        Extract and structure tables from Docling document.

        Args:
            doc: Docling document object

        Returns:
            List of structured ExtractedTable objects
        """
        tables = []
        table_idx = 1

        # Iterate through document tables
        for item, _ in doc.iterate_items():
            if item.label == "table":
                cells = []

                # Extract table grid data if available
                if hasattr(item, 'data') and hasattr(item.data, 'grid'):
                    grid = item.data.grid
                    row_count = len(grid)
                    column_count = len(grid[0]) if grid else 0

                    for row_idx, row in enumerate(grid):
                        for col_idx, cell_content in enumerate(row):
                            cell = TableCell(
                                content=str(cell_content) if cell_content else "",
                                row_index=row_idx,
                                column_index=col_idx,
                                row_span=1,
                                column_span=1,
                                is_header=(row_idx == 0),
                            )
                            cells.append(cell)
                else:
                    # Fallback: use markdown representation
                    row_count = 0
                    column_count = 0

                # Get page number if available
                page_number = item.prov[0].page if hasattr(item, 'prov') and item.prov else None

                table = ExtractedTable(
                    table_id=table_idx,
                    row_count=row_count,
                    column_count=column_count,
                    cells=cells,
                    page_number=page_number,
                )

                tables.append(table)
                table_idx += 1

        logger.debug(f"Extracted {len(tables)} tables using TableFormer")
        return tables

    def _extract_metadata(
        self,
        result: Any,
        parsed: ParsedDocument,
    ) -> PaperMetadata:
        """
        Extract paper metadata from the parsed content.

        Attempts to identify title, authors, abstract, and other
        bibliographic information from the parsed content.

        Args:
            result: Raw Docling extraction result
            parsed: Already processed ParsedDocument

        Returns:
            Extracted PaperMetadata
        """
        import re

        metadata = PaperMetadata()
        doc = result.document

        # Extract title from metadata or first heading
        if hasattr(doc, 'name') and doc.name:
            metadata.title = doc.name
        else:
            # Try to find title in first few items
            for item, _ in doc.iterate_items():
                if item.label in ["title", "section_header"]:
                    text = item.text if hasattr(item, 'text') else str(item)
                    if text and 10 < len(text) < 300:
                        metadata.title = text
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

        return metadata


# Global parser instance
_docling_parser: DoclingPDFParser | None = None


def get_docling_parser() -> DoclingPDFParser:
    """Get or create the global Docling parser instance."""
    global _docling_parser
    if _docling_parser is None:
        _docling_parser = DoclingPDFParser(
            enable_ocr=settings.docling_enable_ocr if hasattr(settings, 'docling_enable_ocr') else True,
        )
    return _docling_parser
