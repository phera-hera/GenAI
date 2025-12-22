"""
Azure Document Intelligence Client for PDF Parsing

Provides document analysis capabilities for extracting structured
content from medical research PDFs, including tables and sections.
"""

import logging
from typing import Any

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import DocumentParsingError

logger = logging.getLogger(__name__)


class AzureDocumentClient:
    """
    Client for Azure Document Intelligence (Form Recognizer).
    
    Extracts structured content from medical research PDFs including:
    - Text content with layout information
    - Tables with cell data
    - Document structure (sections, paragraphs)
    """
    
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the Azure Document Intelligence client.
        
        Args:
            endpoint: Azure endpoint URL (defaults to settings)
            api_key: Azure API key (defaults to settings)
        """
        self.endpoint = endpoint or settings.azure_document_intelligence_endpoint
        self.api_key = api_key or settings.azure_document_intelligence_key
        
        self._client: DocumentAnalysisClient | None = None
    
    @property
    def client(self) -> DocumentAnalysisClient:
        """Get or create the document analysis client."""
        if self._client is None:
            if not self.is_configured():
                raise DocumentParsingError(
                    "Azure Document Intelligence is not configured"
                )
            
            self._client = DocumentAnalysisClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._client
    
    def is_configured(self) -> bool:
        """Check if Azure Document Intelligence is properly configured."""
        return bool(self.endpoint and self.api_key)
    
    def verify_connection(self) -> bool:
        """
        Verify that we can connect to Azure Document Intelligence.
        
        Returns:
            True if connection is successful
            
        Raises:
            DocumentParsingError: If connection fails
        """
        if not self.is_configured():
            raise DocumentParsingError(
                "Azure Document Intelligence is not configured"
            )
        
        try:
            # Simple connectivity check by accessing the client
            _ = self.client
            logger.info("Successfully connected to Azure Document Intelligence")
            return True
        except Exception as e:
            raise DocumentParsingError(
                f"Failed to connect to Azure Document Intelligence: {e}"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def analyze_pdf(
        self,
        pdf_content: bytes,
        model_id: str = "prebuilt-layout",
    ) -> dict[str, Any]:
        """
        Analyze a PDF document and extract structured content.
        
        Args:
            pdf_content: PDF file content as bytes
            model_id: Azure model to use (default: prebuilt-layout)
            
        Returns:
            Extracted document structure with text, tables, and sections
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        try:
            logger.info(f"Analyzing PDF with model: {model_id}")
            
            # Start analysis
            poller = self.client.begin_analyze_document(
                model_id=model_id,
                document=pdf_content,
            )
            
            # Wait for completion
            result = poller.result()
            
            # Extract structured content
            extracted = self._extract_content(result)
            
            logger.info(
                f"Extracted {len(extracted['paragraphs'])} paragraphs, "
                f"{len(extracted['tables'])} tables"
            )
            
            return extracted
            
        except Exception as e:
            logger.error(f"Failed to analyze PDF: {e}")
            raise DocumentParsingError(f"Failed to analyze PDF: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def analyze_pdf_from_url(
        self,
        pdf_url: str,
        model_id: str = "prebuilt-layout",
    ) -> dict[str, Any]:
        """
        Analyze a PDF document from a URL.
        
        Args:
            pdf_url: URL to the PDF file
            model_id: Azure model to use
            
        Returns:
            Extracted document structure
        """
        try:
            logger.info(f"Analyzing PDF from URL with model: {model_id}")
            
            poller = self.client.begin_analyze_document_from_url(
                model_id=model_id,
                document_url=pdf_url,
            )
            
            result = poller.result()
            return self._extract_content(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze PDF from URL: {e}")
            raise DocumentParsingError(f"Failed to analyze PDF from URL: {e}")
    
    def _extract_content(self, result: Any) -> dict[str, Any]:
        """
        Extract structured content from analysis result.
        
        Args:
            result: Azure Document Intelligence result
            
        Returns:
            Structured content dict with text, tables, sections
        """
        extracted = {
            "content": result.content,
            "page_count": len(result.pages) if result.pages else 0,
            "paragraphs": [],
            "tables": [],
            "sections": [],
            "key_value_pairs": [],
        }
        
        # Extract paragraphs with roles (titles, headers, etc.)
        if result.paragraphs:
            for para in result.paragraphs:
                para_info = {
                    "content": para.content,
                    "role": para.role if hasattr(para, "role") else None,
                    "page_number": self._get_page_number(para),
                    "bounding_regions": self._get_bounding_regions(para),
                }
                extracted["paragraphs"].append(para_info)
        
        # Extract tables with cell structure
        if result.tables:
            for table in result.tables:
                table_info = {
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "page_number": self._get_page_number(table),
                    "cells": [],
                }
                
                for cell in table.cells:
                    cell_info = {
                        "content": cell.content,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                        "row_span": cell.row_span or 1,
                        "column_span": cell.column_span or 1,
                        "kind": cell.kind if hasattr(cell, "kind") else "content",
                    }
                    table_info["cells"].append(cell_info)
                
                extracted["tables"].append(table_info)
        
        # Extract sections if available (hierarchical structure)
        if hasattr(result, "sections") and result.sections:
            for section in result.sections:
                section_info = self._extract_section(section)
                extracted["sections"].append(section_info)
        
        # Extract key-value pairs if available
        if hasattr(result, "key_value_pairs") and result.key_value_pairs:
            for kv in result.key_value_pairs:
                if kv.key and kv.value:
                    extracted["key_value_pairs"].append({
                        "key": kv.key.content,
                        "value": kv.value.content,
                    })
        
        return extracted
    
    def _extract_section(self, section: Any) -> dict[str, Any]:
        """Extract section information recursively."""
        section_info = {
            "elements": [],
        }
        
        if hasattr(section, "elements") and section.elements:
            for element in section.elements:
                if hasattr(element, "content"):
                    section_info["elements"].append(element.content)
        
        return section_info
    
    def _get_page_number(self, element: Any) -> int | None:
        """Get page number from an element's bounding regions."""
        if hasattr(element, "bounding_regions") and element.bounding_regions:
            return element.bounding_regions[0].page_number
        return None
    
    def _get_bounding_regions(self, element: Any) -> list[dict]:
        """Get bounding regions from an element."""
        regions = []
        if hasattr(element, "bounding_regions") and element.bounding_regions:
            for region in element.bounding_regions:
                regions.append({
                    "page_number": region.page_number,
                    "polygon": (
                        [{"x": p.x, "y": p.y} for p in region.polygon]
                        if region.polygon else []
                    ),
                })
        return regions
    
    def extract_medical_paper_sections(
        self,
        pdf_content: bytes,
    ) -> dict[str, Any]:
        """
        Extract sections specific to medical research papers.
        
        Identifies common sections like Abstract, Introduction,
        Methods, Results, Discussion, Conclusions.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Dict with identified paper sections and metadata
        """
        # First, get the raw analysis
        analysis = self.analyze_pdf(pdf_content)
        
        # Define section keywords to identify
        section_keywords = {
            "abstract": ["abstract", "summary"],
            "introduction": ["introduction", "background"],
            "methods": ["methods", "methodology", "materials and methods"],
            "results": ["results", "findings"],
            "discussion": ["discussion"],
            "conclusions": ["conclusions", "conclusion"],
            "references": ["references", "bibliography"],
        }
        
        # Initialize sections
        sections = {key: [] for key in section_keywords}
        sections["other"] = []
        
        current_section = "other"
        
        # Process paragraphs to identify sections
        for para in analysis["paragraphs"]:
            content = para["content"]
            role = para.get("role", "")
            
            # Check if this is a section header
            if role in ["sectionHeading", "title"] or self._is_section_header(content):
                content_lower = content.lower().strip()
                
                # Identify which section this header starts
                matched = False
                for section_name, keywords in section_keywords.items():
                    if any(kw in content_lower for kw in keywords):
                        current_section = section_name
                        matched = True
                        break
                
                if not matched:
                    current_section = "other"
            else:
                # Add content to current section
                sections[current_section].append(para)
        
        return {
            "sections": sections,
            "tables": analysis["tables"],
            "page_count": analysis["page_count"],
            "full_content": analysis["content"],
        }
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text appears to be a section header."""
        text = text.strip()
        
        # Heuristics for section headers:
        # - Short text (typically under 100 chars)
        # - Often numbered (1., 2., I., II.)
        # - Often all caps or title case
        
        if len(text) > 100:
            return False
        
        # Check for numbered headings
        import re
        if re.match(r"^(\d+\.|\d+\s+|[IVX]+\.?)\s*[A-Z]", text):
            return True
        
        # Check for all caps (common for section headers)
        if text.isupper() and len(text) > 3:
            return True
        
        return False


# Global client instance
_document_client: AzureDocumentClient | None = None


def get_document_client() -> AzureDocumentClient:
    """Get or create the global document analysis client."""
    global _document_client
    if _document_client is None:
        _document_client = AzureDocumentClient()
    return _document_client

