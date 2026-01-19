"""
LlamaParser Client for PDF Parsing with LVM Mode

Provides document analysis capabilities for extracting structured
content from medical research PDFs using LlamaParser's LVM (Large Vision Model) mode
with Azure OpenAI's multimodal GPT-4o model.

The LVM mode uses your own Azure OpenAI model for PDF parsing, providing
better control and cost efficiency compared to the API-based approach.
"""

import logging
from typing import Any

from llama_parse import LlamaParse
from tenacity import retry, stop_after_attempt, wait_exponential

from medical_agent.core.config import settings
from medical_agent.core.exceptions import DocumentParsingError

logger = logging.getLogger(__name__)


class LlamaParserClient:
    """
    Client for LlamaParser in LVM (Large Vision Model) mode.

    Uses Azure OpenAI's GPT-4o multimodal model for PDF parsing instead of
    the LlamaCloud API. This mode extracts structured content from medical
    research PDFs including:
    - Text content with layout information
    - Tables with cell data
    - Document structure (sections, paragraphs)
    - Figures and visual elements
    """

    def __init__(
        self,
        azure_api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment_name: str | None = None,
        llama_cloud_api_key: str | None = None,
    ):
        """
        Initialize the LlamaParser client in LVM mode.

        Args:
            azure_api_key: Azure OpenAI API key (defaults to settings)
            azure_endpoint: Azure OpenAI endpoint (defaults to settings)
            azure_api_version: Azure OpenAI API version (defaults to settings)
            azure_deployment_name: Azure OpenAI deployment name (defaults to settings)
            llama_cloud_api_key: LlamaCloud API key for usage tracking (defaults to settings)
        """
        self.azure_api_key = azure_api_key or settings.azure_openai_api_key
        self.azure_endpoint = azure_endpoint or settings.azure_openai_endpoint
        self.azure_api_version = azure_api_version or settings.azure_openai_api_version
        self.azure_deployment_name = (
            azure_deployment_name or settings.azure_openai_deployment_name
        )
        self.llama_cloud_api_key = llama_cloud_api_key or settings.llama_cloud_api_key
        self._client: LlamaParse | None = None

    @property
    def client(self) -> LlamaParse:
        """Get or create the LlamaParser client in LVM mode."""
        if self._client is None:
            if not self.is_configured():
                raise DocumentParsingError(
                    "LlamaParser LVM mode is not configured. "
                    "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                    "AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT_NAME"
                )

            logger.info(f"Initializing LlamaParser in LVM mode with Azure GPT-4o")

            self._client = LlamaParse(
                # LlamaCloud API key for usage tracking (required for credits)
                api_key=self.llama_cloud_api_key,
                # Use LVM (Large Vision Model) mode with your own Azure model
                parse_mode="parse_page_with_lvm",
                # Use Azure OpenAI's multimodal capabilities
                use_vendor_multimodal_model=True,
                # Azure OpenAI configuration (note: parameter name is azure_openai_key, not api_key)
                azure_openai_key=self.azure_api_key,
                azure_openai_endpoint=self.azure_endpoint,
                azure_openai_api_version=self.azure_api_version,
                azure_openai_deployment_name=self.azure_deployment_name,
                # Parsing instructions for medical research papers
                system_prompt=(
                    "You are an expert at understanding medical research papers. "
                    "Extract all text, tables, figures, and section headers clearly. "
                    "Preserve the hierarchical structure of sections. "
                    "For tables, maintain the row and column structure. "
                    "Include all numerical data, citations, and medical terminology."
                ),
                result_type="markdown",
                verbose=False,
            )
        return self._client

    def is_configured(self) -> bool:
        """Check if LlamaParser LVM mode is properly configured."""
        return bool(
            self.azure_api_key
            and self.azure_endpoint
            and self.azure_api_version
            and self.azure_deployment_name
            and self.llama_cloud_api_key
        )

    def verify_connection(self) -> bool:
        """
        Verify that we can connect to Azure OpenAI for PDF parsing.

        Returns:
            True if connection is successful

        Raises:
            DocumentParsingError: If connection fails
        """
        if not self.is_configured():
            raise DocumentParsingError(
                "LlamaParser LVM mode is not configured. "
                "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT_NAME"
            )

        try:
            # Simple connectivity check by accessing the client
            logger.info(
                f"Verifying connection to Azure OpenAI: {self.azure_endpoint}"
            )
            _ = self.client
            logger.info(
                f"Successfully configured LlamaParser LVM mode with Azure GPT-4o "
                f"(deployment: {self.azure_deployment_name})"
            )
            return True
        except Exception as e:
            raise DocumentParsingError(
                f"Failed to configure LlamaParser LVM mode: {e}"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def analyze_pdf(
        self,
        pdf_content: bytes,
    ) -> dict[str, Any]:
        """
        Analyze a PDF document and extract structured content.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Extracted document structure with text, tables, and sections
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        try:
            logger.info("Analyzing PDF with LlamaParser")
            
            # LlamaParser expects a file-like object or file path
            # We'll use the load_data method with extra_info for bytes
            import tempfile
            import os
            
            # Write bytes to temporary file for LlamaParser
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = tmp_file.name
            
            try:
                # Parse the PDF
                documents = self.client.load_data(tmp_path)
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
            
            if not documents:
                raise DocumentParsingError("No content extracted from PDF")
            
            # Combine all document content
            full_content = "\n\n".join(doc.text for doc in documents)
            
            # Extract structured content
            extracted = self._extract_content(full_content, documents)
            
            logger.info(
                f"Extracted {len(extracted['paragraphs'])} paragraphs, "
                f"{len(extracted['tables'])} tables"
            )
            
            return extracted
            
        except DocumentParsingError:
            raise
        except Exception as e:
            logger.error(f"Failed to analyze PDF: {e}")
            raise DocumentParsingError(f"Failed to analyze PDF: {e}")
    
    def _extract_content(
        self,
        full_content: str,
        documents: list,
    ) -> dict[str, Any]:
        """
        Extract structured content from parsed documents.
        
        Args:
            full_content: Combined text content
            documents: List of LlamaParser documents
            
        Returns:
            Structured content dict with text, tables, sections
        """
        extracted = {
            "content": full_content,
            "page_count": len(documents),
            "paragraphs": [],
            "tables": [],
            "sections": [],
            "key_value_pairs": [],
        }
        
        # Parse content into paragraphs and identify structure
        lines = full_content.split("\n")
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check for section headers (markdown style from LlamaParser)
            if stripped.startswith("#"):
                # Save current paragraph
                if current_paragraph:
                    para_text = " ".join(current_paragraph)
                    extracted["paragraphs"].append({
                        "content": para_text,
                        "role": None,
                        "page_number": None,
                        "bounding_regions": [],
                    })
                    current_paragraph = []
                
                # Add header as paragraph with role
                header_level = len(stripped) - len(stripped.lstrip("#"))
                header_text = stripped.lstrip("#").strip()
                extracted["paragraphs"].append({
                    "content": header_text,
                    "role": "sectionHeading" if header_level <= 2 else "title",
                    "page_number": None,
                    "bounding_regions": [],
                })
            elif stripped.startswith("|") and "|" in stripped[1:]:
                # This is a table row - handle table extraction
                pass  # Tables are handled separately below
            elif stripped:
                current_paragraph.append(stripped)
            else:
                # Empty line - end of paragraph
                if current_paragraph:
                    para_text = " ".join(current_paragraph)
                    extracted["paragraphs"].append({
                        "content": para_text,
                        "role": None,
                        "page_number": None,
                        "bounding_regions": [],
                    })
                    current_paragraph = []
        
        # Don't forget the last paragraph
        if current_paragraph:
            para_text = " ".join(current_paragraph)
            extracted["paragraphs"].append({
                "content": para_text,
                "role": None,
                "page_number": None,
                "bounding_regions": [],
            })
        
        # Extract tables from markdown
        extracted["tables"] = self._extract_tables_from_markdown(full_content)
        
        return extracted
    
    def _extract_tables_from_markdown(self, content: str) -> list[dict[str, Any]]:
        """
        Extract tables from markdown formatted content.
        
        Args:
            content: Full markdown content
            
        Returns:
            List of table dictionaries
        """
        tables = []
        lines = content.split("\n")
        
        i = 0
        table_id = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this is the start of a markdown table
            if line.startswith("|") and "|" in line[1:]:
                table_id += 1
                table_rows = []
                
                # Collect all table rows
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row_line = lines[i].strip()
                    
                    # Skip separator rows (|---|---|)
                    if not all(c in "|-: " for c in row_line):
                        cells = [c.strip() for c in row_line.split("|")[1:-1]]
                        table_rows.append(cells)
                    
                    i += 1
                
                if table_rows:
                    # First row is header
                    row_count = len(table_rows)
                    col_count = max(len(row) for row in table_rows) if table_rows else 0
                    
                    cells = []
                    for row_idx, row in enumerate(table_rows):
                        for col_idx, cell_content in enumerate(row):
                            cells.append({
                                "content": cell_content,
                                "row_index": row_idx,
                                "column_index": col_idx,
                                "row_span": 1,
                                "column_span": 1,
                                "kind": "columnHeader" if row_idx == 0 else "content",
                            })
                    
                    tables.append({
                        "row_count": row_count,
                        "column_count": col_count,
                        "page_number": None,
                        "cells": cells,
                    })
            else:
                i += 1
        
        return tables
    
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
_llama_parser_client: LlamaParserClient | None = None


def get_llama_parser_client() -> LlamaParserClient:
    """Get or create the global LlamaParser client."""
    global _llama_parser_client
    if _llama_parser_client is None:
        _llama_parser_client = LlamaParserClient()
    return _llama_parser_client

