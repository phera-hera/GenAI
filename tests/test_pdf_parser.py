"""
Tests for PDF parsing functionality.

Tests the PDF parsers for medical research paper extraction,
including section identification, table extraction, and metadata parsing.
"""

import pytest
from unittest.mock import MagicMock, patch

from medical_agent.ingestion.parsers import (
    DocumentSection,
    ExtractedTable,
    MedicalPDFParser,
    PaperMetadata,
    ParsedDocument,
    PDFParserFacade,
    SectionType,
    TableCell,
    get_parser_facade,
    parse_pdf,
)


class TestDocumentResult:
    """Tests for document result data classes."""
    
    def test_table_cell_creation(self):
        """Test TableCell creation and string representation."""
        cell = TableCell(
            content="Test content",
            row_index=0,
            column_index=1,
            is_header=True,
        )
        
        assert cell.content == "Test content"
        assert cell.row_index == 0
        assert cell.column_index == 1
        assert cell.is_header is True
        assert str(cell) == "Test content"
    
    def test_extracted_table_to_markdown(self):
        """Test table conversion to markdown."""
        cells = [
            TableCell("Name", 0, 0, is_header=True),
            TableCell("Value", 0, 1, is_header=True),
            TableCell("pH", 1, 0),
            TableCell("4.2", 1, 1),
        ]
        
        table = ExtractedTable(
            table_id=1,
            row_count=2,
            column_count=2,
            cells=cells,
            caption="pH Values",
        )
        
        markdown = table.to_markdown()
        
        assert "**pH Values**" in markdown
        assert "| Name | Value |" in markdown
        assert "| --- | --- |" in markdown
        assert "| pH | 4.2 |" in markdown
    
    def test_extracted_table_to_dict(self):
        """Test table conversion to dictionary."""
        cells = [TableCell("Test", 0, 0)]
        table = ExtractedTable(
            table_id=1,
            row_count=1,
            column_count=1,
            cells=cells,
        )
        
        result = table.to_dict()
        
        assert result["table_id"] == 1
        assert result["row_count"] == 1
        assert "cells" in result
        assert "markdown" in result
    
    def test_document_section_full_content(self):
        """Test section content aggregation."""
        subsection = DocumentSection(
            section_type=SectionType.OTHER,
            title="Sub-analysis",
            content="Detailed analysis here.",
        )
        
        section = DocumentSection(
            section_type=SectionType.RESULTS,
            title="Results",
            content="Main results text.",
            subsections=[subsection],
        )
        
        full = section.full_content
        
        assert "Main results text" in full
        assert "Sub-analysis" in full
        assert "Detailed analysis" in full
    
    def test_paper_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = PaperMetadata(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            doi="10.1234/test",
            keywords=["vaginal health", "pH"],
        )
        
        result = metadata.to_dict()
        
        assert result["title"] == "Test Paper"
        assert len(result["authors"]) == 2
        assert result["doi"] == "10.1234/test"
    
    def test_parsed_document_get_section(self):
        """Test section retrieval by type."""
        abstract = DocumentSection(
            section_type=SectionType.ABSTRACT,
            title="Abstract",
            content="This is the abstract.",
        )
        methods = DocumentSection(
            section_type=SectionType.METHODS,
            title="Methods",
            content="Research methods.",
        )
        
        doc = ParsedDocument(
            full_text="Full document text",
            sections=[abstract, methods],
        )
        
        assert doc.get_section(SectionType.ABSTRACT) == abstract
        assert doc.get_section(SectionType.METHODS) == methods
        assert doc.get_section(SectionType.DISCUSSION) is None
    
    def test_parsed_document_get_abstract(self):
        """Test abstract retrieval from various sources."""
        # Test from metadata
        doc = ParsedDocument(
            full_text="Text",
            metadata=PaperMetadata(abstract="Metadata abstract"),
        )
        assert doc.get_abstract() == "Metadata abstract"
        
        # Test from section
        abstract_section = DocumentSection(
            section_type=SectionType.ABSTRACT,
            title="Abstract",
            content="Section abstract",
        )
        doc2 = ParsedDocument(
            full_text="Text",
            sections=[abstract_section],
        )
        assert doc2.get_abstract() == "Section abstract"
    
    def test_parsed_document_summary(self):
        """Test document summary generation."""
        doc = ParsedDocument(
            full_text="Text",
            page_count=10,
            sections=[
                DocumentSection(SectionType.ABSTRACT, "Abstract", "Content"),
            ],
            tables=[
                ExtractedTable(1, 2, 2, []),
            ],
            metadata=PaperMetadata(title="Test Paper"),
        )
        
        summary = doc.summary()
        
        assert "Test Paper" in summary
        assert "10" in summary
        assert "1" in summary  # sections count
        assert "Success" in summary


class TestMedicalPDFParser:
    """Tests for the Azure-based PDF parser."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = MedicalPDFParser()
        assert parser is not None
    
    def test_section_pattern_matching(self):
        """Test section header identification."""
        parser = MedicalPDFParser()
        
        # Test various header formats
        assert parser._identify_section_type("Abstract", "sectionHeading") == SectionType.ABSTRACT
        assert parser._identify_section_type("1. Introduction", "sectionHeading") == SectionType.INTRODUCTION
        assert parser._identify_section_type("Materials and Methods", "sectionHeading") == SectionType.METHODS
        assert parser._identify_section_type("3. RESULTS", "sectionHeading") == SectionType.RESULTS
        assert parser._identify_section_type("Discussion", "sectionHeading") == SectionType.DISCUSSION
        assert parser._identify_section_type("Conclusions", "sectionHeading") == SectionType.CONCLUSION
        assert parser._identify_section_type("References", "sectionHeading") == SectionType.REFERENCES
    
    def test_header_heuristics(self):
        """Test header detection heuristics."""
        parser = MedicalPDFParser()
        
        # Short text should be considered as potential header
        assert parser._looks_like_header("METHODS") is True
        assert parser._looks_like_header("1. Introduction") is True
        assert parser._looks_like_header("II. Background") is True
        
        # Long text should not be a header
        long_text = "This is a very long paragraph that contains " * 5
        assert parser._looks_like_header(long_text) is False
    
    @patch.object(MedicalPDFParser, 'azure_client')
    def test_parse_with_mock(self, mock_client):
        """Test parsing with mocked Azure client."""
        # Setup mock response
        mock_result = {
            "content": "Full text of the paper",
            "page_count": 5,
            "paragraphs": [
                {"content": "Paper Title", "role": "title"},
                {"content": "Abstract", "role": "sectionHeading"},
                {"content": "This is the abstract content.", "role": ""},
                {"content": "Methods", "role": "sectionHeading"},
                {"content": "Research methodology description.", "role": ""},
            ],
            "tables": [
                {
                    "row_count": 2,
                    "column_count": 2,
                    "page_number": 3,
                    "cells": [
                        {"content": "pH", "row_index": 0, "column_index": 0, "kind": "columnHeader"},
                        {"content": "Count", "row_index": 0, "column_index": 1, "kind": "columnHeader"},
                        {"content": "4.2", "row_index": 1, "column_index": 0},
                        {"content": "50", "row_index": 1, "column_index": 1},
                    ],
                }
            ],
            "key_value_pairs": [],
        }
        
        mock_client.analyze_pdf.return_value = mock_result
        mock_client.is_configured.return_value = True
        
        parser = MedicalPDFParser()
        parser._azure_client = mock_client
        
        result = parser.parse(b"fake pdf content", source_file="test.pdf")
        
        assert result.is_successful
        assert result.page_count == 5
        assert len(result.tables) == 1
        assert result.tables[0].row_count == 2
        assert result.metadata.source_file == "test.pdf"


class TestPDFParserFacade:
    """Tests for the parser facade."""
    
    def test_facade_initialization(self):
        """Test facade initialization."""
        facade = PDFParserFacade()
        assert facade.prefer_azure is True
        assert facade.allow_fallback is True
    
    def test_facade_with_no_parsers(self):
        """Test facade when no parsers are available."""
        facade = PDFParserFacade(prefer_azure=False, allow_fallback=False)
        
        # Mock both parsers as unavailable
        with patch.object(facade.azure_parser, 'is_available', return_value=False):
            with patch.object(facade.fallback_parser, 'is_available', return_value=False):
                result = facade.parse(b"pdf content", source_file="test.pdf")
                
                assert not result.is_successful
                assert len(result.parsing_errors) > 0
    
    def test_get_parser_name(self):
        """Test parser name resolution."""
        facade = PDFParserFacade()
        
        with patch.object(facade.azure_parser, 'is_available', return_value=True):
            assert facade.get_parser_name() == "azure"
        
        with patch.object(facade.azure_parser, 'is_available', return_value=False):
            with patch.object(facade.fallback_parser, 'is_available', return_value=True):
                assert facade.get_parser_name() == "fallback"


class TestSectionType:
    """Tests for section type enum."""
    
    def test_section_types(self):
        """Test all section types exist."""
        assert SectionType.ABSTRACT == "abstract"
        assert SectionType.INTRODUCTION == "introduction"
        assert SectionType.METHODS == "methods"
        assert SectionType.RESULTS == "results"
        assert SectionType.DISCUSSION == "discussion"
        assert SectionType.CONCLUSION == "conclusion"
        assert SectionType.REFERENCES == "references"
        assert SectionType.OTHER == "other"


class TestConvenienceFunction:
    """Tests for the parse_pdf convenience function."""
    
    def test_parse_pdf_function_exists(self):
        """Test that parse_pdf is importable and callable."""
        assert callable(parse_pdf)
    
    def test_get_parser_facade_singleton(self):
        """Test that get_parser_facade returns same instance."""
        # Note: This test might not work in all scenarios due to module-level state
        facade1 = get_parser_facade()
        facade2 = get_parser_facade()
        assert facade1 is facade2

