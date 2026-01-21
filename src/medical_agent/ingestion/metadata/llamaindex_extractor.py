"""
LlamaIndex Medical Metadata Extractor Wrapper

Wraps the existing MedicalMetadataExtractor to work with LlamaIndex's
IngestionPipeline by implementing the BaseExtractor interface.

Extracts document-level medical metadata once and stamps it onto all nodes.
"""

import logging
from typing import Any, Sequence

from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode, TextNode

from medical_agent.ingestion.metadata.extractor import MedicalMetadataExtractor
from medical_agent.ingestion.parsers.document_result import ParsedDocument, PaperMetadata, SectionType

logger = logging.getLogger(__name__)


class LlamaIndexMedicalMetadataExtractor(BaseExtractor):
    """
    LlamaIndex-compatible wrapper for MedicalMetadataExtractor.

    Extracts medical metadata once for the entire document and stamps
    it onto all nodes in the metadata["extracted_metadata"] field.

    The extractor reconstructs a simplified ParsedDocument from the nodes
    to leverage the existing extraction logic, then applies the results
    to all nodes.
    """

    def __init__(
        self,
        extractor: MedicalMetadataExtractor | None = None,
        extract_table_summaries: bool = True,
        **kwargs,
    ):
        """
        Initialize the LlamaIndex metadata extractor wrapper.

        Args:
            extractor: Optional custom MedicalMetadataExtractor instance
            extract_table_summaries: Whether to extract table summaries
            **kwargs: Additional arguments for BaseExtractor
        """
        super().__init__(**kwargs)
        # Store as private attributes to avoid Pydantic validation
        object.__setattr__(self, "_extractor", extractor or MedicalMetadataExtractor())
        object.__setattr__(self, "_extract_table_summaries", extract_table_summaries)
        object.__setattr__(self, "_document_cache", {})

    def _reconstruct_parsed_document(self, nodes: Sequence[BaseNode]) -> ParsedDocument:
        """
        Reconstruct a ParsedDocument from LlamaIndex nodes.

        This creates a simplified ParsedDocument that has enough context
        for the MedicalMetadataExtractor to work with.

        Args:
            nodes: List of LlamaIndex nodes

        Returns:
            ParsedDocument with combined text and metadata
        """
        # Combine all text content from nodes
        full_text_parts = []
        abstract_text = None
        methods_text = None
        tables = []
        title = None

        for node in nodes:
            if not isinstance(node, TextNode):
                continue

            content = node.get_content()
            full_text_parts.append(content)

            # Extract section-specific content
            section_type = node.metadata.get("section_type", "").lower()
            chunk_type = node.metadata.get("chunk_type", "").lower()

            if "abstract" in section_type or chunk_type == "abstract":
                abstract_text = content
            elif "method" in section_type:
                methods_text = content
            elif chunk_type == "table":
                # Note: We can't fully reconstruct table structure from nodes,
                # but we can pass the table content
                tables.append(content)

            # Get title from first node if available
            if title is None and "title" in node.metadata:
                title = node.metadata.get("title")

        full_text = "\n\n".join(full_text_parts)

        # Create paper metadata
        paper_metadata = PaperMetadata(
            title=title or "Unknown",
            abstract=abstract_text,
        )

        # Create a simplified ParsedDocument
        # We create a minimal document with the essential fields
        parsed_doc = ParsedDocument(
            full_text=full_text,
            metadata=paper_metadata,
        )

        # If we found abstract or methods text, add them as pseudo-sections
        # This helps the extractor get better context
        if abstract_text:
            from medical_agent.ingestion.parsers.document_result import DocumentSection
            abstract_section = DocumentSection(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                content=abstract_text,
            )
            parsed_doc.sections.append(abstract_section)

        if methods_text:
            from medical_agent.ingestion.parsers.document_result import DocumentSection
            methods_section = DocumentSection(
                section_type=SectionType.METHODS,
                title="Methods",
                content=methods_text,
            )
            parsed_doc.sections.append(methods_section)

        return parsed_doc

    async def aextract(
        self, nodes: Sequence[BaseNode]
    ) -> list[dict[str, Any]]:
        """
        Extract medical metadata from nodes.

        This method is called by LlamaIndex's IngestionPipeline. It extracts
        document-level medical metadata once and applies it to all nodes.

        Args:
            nodes: List of nodes from a single document

        Returns:
            List of metadata dicts, one per node
        """
        if not nodes:
            return []

        try:
            # Get document ID from first node (all nodes from same document should have same ID)
            doc_id = nodes[0].ref_doc_id or "unknown"

            # Check if we've already extracted metadata for this document
            if doc_id in self._document_cache:
                logger.debug(f"Using cached metadata for document {doc_id}")
                cached_metadata = self._document_cache[doc_id]
            else:
                logger.info(f"Extracting metadata for document {doc_id} ({len(nodes)} nodes)")

                # Reconstruct ParsedDocument from nodes
                parsed_doc = self._reconstruct_parsed_document(nodes)

                # Extract medical metadata using existing extractor
                extracted_metadata = await self._extractor.extract(
                    parsed=parsed_doc,
                    extract_table_summaries=self._extract_table_summaries,
                )

                # Convert to dict format for storage
                cached_metadata = {
                    "extracted_metadata": {
                        "ethnicities": extracted_metadata.ethnicities,
                        "diagnoses": extracted_metadata.diagnoses,
                        "symptoms": extracted_metadata.symptoms,
                        "menstrual_status": extracted_metadata.menstrual_status,
                        "birth_control": extracted_metadata.birth_control,
                        "hormone_therapy": extracted_metadata.hormone_therapy,
                        "fertility_treatments": extracted_metadata.fertility_treatments,
                        "age_mentioned": extracted_metadata.age_mentioned,
                        "age_range": extracted_metadata.age_range,
                        "confidence": extracted_metadata.confidence,
                    },
                    "table_summaries": extracted_metadata.table_summaries,
                }

                # Cache for future nodes from same document
                self._document_cache[doc_id] = cached_metadata

                logger.info(
                    f"Extracted metadata for document {doc_id}: "
                    f"{sum(len(v) if isinstance(v, list) else 0 for v in cached_metadata['extracted_metadata'].values())} terms, "
                    f"confidence: {cached_metadata['extracted_metadata']['confidence']:.2f}"
                )

            # Apply metadata to all nodes
            result = []
            for node in nodes:
                node_metadata = cached_metadata["extracted_metadata"].copy()

                # Add table summary if this is a table chunk
                chunk_type = node.metadata.get("chunk_type")
                if chunk_type == "table":
                    table_id = node.metadata.get("table_id")
                    if table_id is not None and table_id in cached_metadata["table_summaries"]:
                        node_metadata["table_summary"] = cached_metadata["table_summaries"][table_id]

                result.append(node_metadata)

            return result

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}", exc_info=True)
            # Return empty metadata for all nodes on failure
            return [{} for _ in nodes]

    def clear_cache(self):
        """Clear the document metadata cache."""
        self._document_cache.clear()
        logger.debug("Cleared metadata cache")
