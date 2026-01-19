"""
Medical Metadata Extractor

Orchestrates extraction and normalization of medical metadata from parsed documents.
Runs after parsing, before chunking, and stamps metadata onto every chunk.
"""

import logging
from typing import Any

from medical_agent.core.config import settings
from medical_agent.ingestion.parsers.document_result import ParsedDocument, SectionType

from .llm_client import MetadataLLMClient, get_metadata_llm_client
from .normalizer import TermNormalizer, get_term_normalizer
from .types import ExtractedMetadata

logger = logging.getLogger(__name__)


class MedicalMetadataExtractor:
    """
    Extracts and normalizes medical metadata from parsed research papers.

    Pipeline:
    1. Extract text from abstract + keywords + methods sections
    2. Use LLM to extract medical terms (ethnicities, diagnoses, symptoms, etc.)
    3. Normalize terms to standardized dropdown values
    4. Extract table summaries if tables are present
    5. Return complete ExtractedMetadata with all categories

    The extracted metadata is stamped onto every chunk in chunk_metadata.
    """

    def __init__(
        self,
        llm_client: MetadataLLMClient | None = None,
        normalizer: TermNormalizer | None = None,
    ):
        """
        Initialize the metadata extractor.

        Args:
            llm_client: Optional LLM client for extraction
            normalizer: Optional term normalizer
        """
        self.llm_client = llm_client or get_metadata_llm_client()
        self.normalizer = normalizer or get_term_normalizer()

    def _extract_relevant_text(self, parsed: ParsedDocument) -> str:
        """
        Extract relevant text for metadata extraction.

        Prioritizes:
        1. Abstract
        2. Keywords from metadata
        3. Methods section (first 3 chunks if sections not properly identified)

        Args:
            parsed: Parsed document

        Returns:
            Combined text for metadata extraction
        """
        parts = []

        # 1. Abstract
        abstract = parsed.get_abstract()
        if abstract:
            parts.append(f"ABSTRACT:\n{abstract}")

        # 2. Keywords
        if parsed.metadata.keywords:
            keywords_text = ", ".join(parsed.metadata.keywords)
            parts.append(f"KEYWORDS: {keywords_text}")

        # 3. Methods section (or introduction if methods not found)
        methods_section = parsed.get_section(SectionType.METHODS)
        if methods_section and methods_section.content:
            # Limit methods section to avoid token overflow
            methods_text = methods_section.content[:2000]
            parts.append(f"METHODS:\n{methods_text}")
        else:
            # Fallback: use introduction or first section
            intro_section = parsed.get_section(SectionType.INTRODUCTION)
            if intro_section and intro_section.content:
                intro_text = intro_section.content[:1500]
                parts.append(f"INTRODUCTION:\n{intro_text}")

        # Combine all parts
        combined_text = "\n\n".join(parts)

        # If still empty, use first part of full text
        if not combined_text.strip():
            combined_text = parsed.full_text[:3000]

        return combined_text

    def _prepare_tables_for_summary(
        self, parsed: ParsedDocument
    ) -> list[dict[str, Any]]:
        """
        Prepare tables for LLM summarization.

        Args:
            parsed: Parsed document

        Returns:
            List of table dicts with table_id and markdown
        """
        tables = []

        for table in parsed.tables:
            tables.append({
                "table_id": table.table_id,
                "markdown": table.to_markdown(),
                "caption": table.caption,
            })

        return tables

    async def extract(
        self,
        parsed: ParsedDocument,
        extract_table_summaries: bool = True,
    ) -> ExtractedMetadata:
        """
        Extract and normalize medical metadata from a parsed document.

        Args:
            parsed: Parsed document from Docling or other parser
            extract_table_summaries: Whether to generate table summaries

        Returns:
            ExtractedMetadata with all categories normalized
        """
        # Check if metadata extraction is enabled
        if not getattr(settings, "metadata_extraction_enabled", True):
            logger.info("Metadata extraction is disabled, returning empty metadata")
            return ExtractedMetadata()

        logger.info(f"Extracting metadata from paper: {parsed.metadata.title or 'Unknown'}")

        # Step 1: Extract relevant text
        text = self._extract_relevant_text(parsed)

        if not text.strip():
            logger.warning("No text available for metadata extraction")
            return ExtractedMetadata()

        # Step 2: Extract medical terms using LLM
        raw_metadata = await self.llm_client.extract_metadata(
            text=text,
            source_description=parsed.metadata.title or "research paper",
        )

        # Step 3: Normalize terms to dropdown values
        normalized_metadata = self.normalizer.normalize(raw_metadata)

        # Step 4: Extract table summaries if requested
        if extract_table_summaries and parsed.tables:
            try:
                tables_data = self._prepare_tables_for_summary(parsed)
                table_summaries = await self.llm_client.summarize_tables(tables_data)

                # Add table summaries to metadata
                normalized_metadata.table_summaries = {
                    tid: summary.summary for tid, summary in table_summaries.items()
                }

                logger.info(f"Extracted {len(table_summaries)} table summaries")

            except Exception as e:
                logger.warning(f"Failed to extract table summaries: {e}")
                # Continue without table summaries

        # Check confidence threshold
        confidence_threshold = getattr(settings, "metadata_extraction_confidence_threshold", 0.0)
        if normalized_metadata.confidence < confidence_threshold:
            logger.warning(
                f"Metadata extraction confidence ({normalized_metadata.confidence:.2f}) "
                f"below threshold ({confidence_threshold:.2f})"
            )

        # Log extraction summary
        total_terms = sum(
            len(getattr(normalized_metadata, field))
            for field in [
                "ethnicities",
                "diagnoses",
                "symptoms",
                "menstrual_status",
                "birth_control",
                "hormone_therapy",
                "fertility_treatments",
            ]
        )

        logger.info(
            f"Metadata extraction complete: {total_terms} total terms, "
            f"confidence: {normalized_metadata.confidence:.2f}, "
            f"empty: {normalized_metadata.is_empty()}"
        )

        return normalized_metadata

    def extract_sync(
        self,
        parsed: ParsedDocument,
        extract_table_summaries: bool = True,
    ) -> ExtractedMetadata:
        """
        Synchronous wrapper for extract method.

        Args:
            parsed: Parsed document
            extract_table_summaries: Whether to generate table summaries

        Returns:
            ExtractedMetadata with all categories normalized
        """
        import asyncio

        return asyncio.run(self.extract(parsed, extract_table_summaries))


# Global extractor instance
_extractor: MedicalMetadataExtractor | None = None


def get_metadata_extractor() -> MedicalMetadataExtractor:
    """Get or create the global metadata extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = MedicalMetadataExtractor()
    return _extractor
