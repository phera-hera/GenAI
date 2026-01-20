"""
Data Classes for Parsed Document Results

Provides structured representations of parsed medical research papers,
including extracted text, tables, sections, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SectionType(str, Enum):
    """Types of sections found in medical research papers."""
    
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    APPENDIX = "appendix"
    OTHER = "other"


@dataclass
class TableCell:
    """Represents a single cell in an extracted table."""
    
    content: str
    row_index: int
    column_index: int
    row_span: int = 1
    column_span: int = 1
    is_header: bool = False
    
    def __str__(self) -> str:
        return self.content


@dataclass
class ExtractedTable:
    """
    Represents a table extracted from a medical paper.
    
    Tables in medical papers often contain critical data like
    study results, patient demographics, or statistical analyses.
    """
    
    table_id: int
    row_count: int
    column_count: int
    cells: list[TableCell]
    page_number: int | None = None
    caption: str | None = None
    
    def to_html(self) -> str:
        """
        Convert the table to HTML format for better structure preservation.

        Returns:
            HTML representation of the table
        """
        if not self.cells:
            return ""

        # Build a 2D grid
        grid: list[list[str]] = [
            ["" for _ in range(self.column_count)]
            for _ in range(self.row_count)
        ]

        # Fill the grid with cell contents
        for cell in self.cells:
            if cell.row_index < self.row_count and cell.column_index < self.column_count:
                grid[cell.row_index][cell.column_index] = cell.content

        # Generate HTML
        html_parts = []

        # Add caption if present
        if self.caption:
            html_parts.append(f"<caption>{self.caption}</caption>")

        # Add table rows
        html_parts.append("<table>")
        for row_idx, row in enumerate(grid):
            html_parts.append("  <tr>")
            # First row is header
            tag = "th" if row_idx == 0 else "td"
            for cell_content in row:
                html_parts.append(f"    <{tag}>{cell_content or ''}</{tag}>")
            html_parts.append("  </tr>")
        html_parts.append("</table>")

        return "\n".join(html_parts)

    def to_markdown(self) -> str:
        """
        Convert the table to markdown format.

        Returns:
            Markdown representation of the table
        """
        if not self.cells:
            return ""

        # Build a 2D grid
        grid: list[list[str]] = [
            ["" for _ in range(self.column_count)]
            for _ in range(self.row_count)
        ]

        # Fill the grid with cell contents
        for cell in self.cells:
            if cell.row_index < self.row_count and cell.column_index < self.column_count:
                grid[cell.row_index][cell.column_index] = cell.content

        # Generate markdown
        lines = []

        # Add caption if present
        if self.caption:
            lines.append(f"**{self.caption}**\n")

        for row_idx, row in enumerate(grid):
            # Create row
            row_text = "| " + " | ".join(cell or "" for cell in row) + " |"
            lines.append(row_text)

            # Add header separator after first row
            if row_idx == 0:
                separator = "| " + " | ".join("---" for _ in row) + " |"
                lines.append(separator)

        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert table to dictionary representation."""
        return {
            "table_id": self.table_id,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "page_number": self.page_number,
            "caption": self.caption,
            "cells": [
                {
                    "content": cell.content,
                    "row": cell.row_index,
                    "col": cell.column_index,
                    "row_span": cell.row_span,
                    "col_span": cell.column_span,
                    "is_header": cell.is_header,
                }
                for cell in self.cells
            ],
            "markdown": self.to_markdown(),
        }


@dataclass
class DocumentSection:
    """
    Represents a section of a medical research paper.
    
    Medical papers typically follow a structured format:
    Abstract -> Introduction -> Methods -> Results -> Discussion -> Conclusion
    """
    
    section_type: SectionType
    title: str | None
    content: str
    paragraphs: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    subsections: list["DocumentSection"] = field(default_factory=list)
    
    @property
    def full_content(self) -> str:
        """Get the complete content including subsections."""
        parts = [self.content] if self.content else []
        
        for subsection in self.subsections:
            if subsection.title:
                parts.append(f"\n### {subsection.title}\n")
            parts.append(subsection.full_content)
        
        return "\n\n".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert section to dictionary representation."""
        return {
            "section_type": self.section_type.value,
            "title": self.title,
            "content": self.content,
            "page_numbers": self.page_numbers,
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class PaperMetadata:
    """
    Metadata extracted from a medical research paper.
    
    Stores bibliographic information that helps with
    citations and paper identification.
    """
    
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    journal: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    keywords: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    
    # Processing metadata
    source_file: str | None = None
    file_hash: str | None = None
    parsed_at: datetime | None = None
    parser_version: str = "1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "journal": self.journal,
            "publication_year": self.publication_year,
            "doi": self.doi,
            "keywords": self.keywords,
            "affiliations": self.affiliations,
            "source_file": self.source_file,
            "file_hash": self.file_hash,
            "parsed_at": self.parsed_at.isoformat() if self.parsed_at else None,
            "parser_version": self.parser_version,
        }


@dataclass
class ParsedDocument:
    """
    Complete representation of a parsed medical research paper.
    
    Contains all extracted content including:
    - Full text content
    - Identified sections
    - Extracted tables
    - Paper metadata
    - Page-level information
    """
    
    # Core content
    full_text: str
    sections: list[DocumentSection] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    metadata: PaperMetadata = field(default_factory=PaperMetadata)
    
    # Document structure
    page_count: int = 0
    
    # Raw extraction data (for debugging/reprocessing)
    raw_paragraphs: list[dict[str, Any]] = field(default_factory=list)
    
    # Parsing status
    is_successful: bool = True
    parsing_errors: list[str] = field(default_factory=list)
    parsing_warnings: list[str] = field(default_factory=list)
    
    def get_section(self, section_type: SectionType) -> DocumentSection | None:
        """
        Get a specific section by type.
        
        Args:
            section_type: The type of section to retrieve
            
        Returns:
            The section if found, None otherwise
        """
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
    
    def get_abstract(self) -> str | None:
        """Get the abstract text."""
        # First try from metadata
        if self.metadata.abstract:
            return self.metadata.abstract
        
        # Then try from sections
        abstract_section = self.get_section(SectionType.ABSTRACT)
        if abstract_section:
            return abstract_section.content
        
        return None
    
    def get_tables_as_text(self) -> str:
        """
        Get all tables formatted as text for embedding.
        
        Returns:
            Concatenated markdown representation of all tables
        """
        table_texts = []
        for table in self.tables:
            table_md = table.to_markdown()
            if table_md:
                table_texts.append(f"Table {table.table_id}:\n{table_md}")
        
        return "\n\n".join(table_texts)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the entire parsed document to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "full_text": self.full_text,
            "sections": [s.to_dict() for s in self.sections],
            "tables": [t.to_dict() for t in self.tables],
            "page_count": self.page_count,
            "is_successful": self.is_successful,
            "parsing_errors": self.parsing_errors,
            "parsing_warnings": self.parsing_warnings,
        }
    
    def summary(self) -> str:
        """Generate a summary of the parsed document."""
        lines = [
            f"Title: {self.metadata.title or 'Unknown'}",
            f"Pages: {self.page_count}",
            f"Sections: {len(self.sections)}",
            f"Tables: {len(self.tables)}",
            f"Status: {'Success' if self.is_successful else 'Failed'}",
        ]
        
        if self.parsing_errors:
            lines.append(f"Errors: {len(self.parsing_errors)}")
        
        if self.parsing_warnings:
            lines.append(f"Warnings: {len(self.parsing_warnings)}")
        
        return "\n".join(lines)

