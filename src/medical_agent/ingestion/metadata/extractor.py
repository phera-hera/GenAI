"""
LlamaIndex Medical Metadata Extractor

Extracts document-level medical metadata from LlamaIndex nodes by implementing
the BaseExtractor interface. Stamps medical metadata onto all nodes for filtering.
"""

import logging
from typing import Any, Sequence

from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode, TextNode

from medical_agent.core.config import settings
from medical_agent.ingestion.metadata.llm_client import MetadataLLMClient, get_metadata_llm_client
from medical_agent.ingestion.metadata.normalizer import TermNormalizer, get_term_normalizer
from medical_agent.ingestion.parsers.document_result import ParsedDocument, PaperMetadata, SectionType

logger = logging.getLogger(__name__)


class MedicalMetadataExtractor(BaseExtractor):
    """
    LlamaIndex-compatible medical metadata extractor.

    Extracts medical metadata once for the entire document and stamps
    it onto all nodes in the metadata["extracted_metadata"] field.

    Uses LLM client and term normalizer directly to extract and normalize
    medical terms (ethnicities, diagnoses, symptoms, etc.) from document text.
    """

    def __init__(
        self,
        llm_client: MetadataLLMClient | None = None,
        normalizer: TermNormalizer | None = None,
        extract_table_summaries: bool = True,
        **kwargs,
    ):
        """
        Initialize the LlamaIndex metadata extractor.

        Args:
            llm_client: Optional custom LLM client for extraction
            normalizer: Optional custom term normalizer
            extract_table_summaries: Whether to extract table summaries
            **kwargs: Additional arguments for BaseExtractor
        """
        super().__init__(**kwargs)
        # Store as private attributes to avoid Pydantic validation
        object.__setattr__(self, "_llm_client", llm_client or get_metadata_llm_client())
        object.__setattr__(self, "_normalizer", normalizer or get_term_normalizer())
        object.__setattr__(self, "_extract_table_summaries", extract_table_summaries)
        object.__setattr__(self, "_document_cache", {})

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

                # Check if metadata extraction is enabled
                if not getattr(settings, "metadata_extraction_enabled", True):
                    logger.info("Metadata extraction is disabled, returning empty metadata")
                    cached_metadata = {
                        "extracted_metadata": {
                            "ethnicities": [],
                            "diagnoses": [],
                            "symptoms": [],
                            "menstrual_status": [],
                            "birth_control": [],
                            "hormone_therapy": [],
                            "fertility_treatments": [],
                            "age_mentioned": False,
                            "age_range": None,
                            "confidence": 0.0,
                        },
                        "table_summaries": {},
                    }
                else:
                    # Reconstruct ParsedDocument from nodes
                    parsed_doc = self._reconstruct_parsed_document(nodes)

                    # Extract relevant text from the document
                    text = self._extract_relevant_text(parsed_doc)

                    if not text.strip():
                        logger.warning("No text available for metadata extraction")
                        cached_metadata = {
                            "extracted_metadata": {
                                "ethnicities": [],
                                "diagnoses": [],
                                "symptoms": [],
                                "menstrual_status": [],
                                "birth_control": [],
                                "hormone_therapy": [],
                                "fertility_treatments": [],
                                "age_mentioned": False,
                                "age_range": None,
                                "confidence": 0.0,
                            },
                            "table_summaries": {},
                        }
                    else:
                        # Extract medical terms using LLM (direct call)
                        raw_metadata = await self._llm_client.extract_metadata(
                            text=text,
                            source_description=parsed_doc.metadata.title or "research paper",
                        )

                        # Normalize terms to dropdown values (direct call)
                        normalized_metadata = self._normalizer.normalize(raw_metadata)

                        # Extract table summaries if requested
                        table_summaries = {}
                        if self._extract_table_summaries and parsed_doc.tables:
                            try:
                                tables_data = [
                                    {
                                        "table_id": table.table_id,
                                        "markdown": table.to_markdown(),
                                        "caption": table.caption,
                                    }
                                    for table in parsed_doc.tables
                                ]
                                table_summary_results = await self._llm_client.summarize_tables(tables_data)
                                table_summaries = {
                                    tid: summary.summary for tid, summary in table_summary_results.items()
                                }
                                logger.info(f"Extracted {len(table_summaries)} table summaries")
                            except Exception as e:
                                logger.warning(f"Failed to extract table summaries: {e}")

                        # Convert to dict format for storage
                        cached_metadata = {
                            "extracted_metadata": {
                                "ethnicities": normalized_metadata.ethnicities,
                                "diagnoses": normalized_metadata.diagnoses,
                                "symptoms": normalized_metadata.symptoms,
                                "menstrual_status": normalized_metadata.menstrual_status,
                                "birth_control": normalized_metadata.birth_control,
                                "hormone_therapy": normalized_metadata.hormone_therapy,
                                "fertility_treatments": normalized_metadata.fertility_treatments,
                                "age_mentioned": normalized_metadata.age_mentioned,
                                "age_range": normalized_metadata.age_range,
                                "confidence": normalized_metadata.confidence,
                            },
                            "table_summaries": table_summaries,
                        }

                        # Log extraction summary
                        total_terms = sum(
                            len(v) if isinstance(v, list) else 0
                            for v in cached_metadata["extracted_metadata"].values()
                        )
                        logger.info(
                            f"Extracted metadata for document {doc_id}: "
                            f"{total_terms} terms, "
                            f"confidence: {cached_metadata['extracted_metadata']['confidence']:.2f}"
                        )

                # Cache for future nodes from same document
                self._document_cache[doc_id] = cached_metadata

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
