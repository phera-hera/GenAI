"""LlamaIndex BaseExtractor for stamping medical metadata onto nodes."""

import logging
from typing import Any, Sequence

from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode, TextNode

from medical_agent.core.config import settings
from medical_agent.ingestion.metadata.llm_client import MetadataLLMClient, get_metadata_llm_client
from medical_agent.ingestion.metadata.normalizer import TermNormalizer, get_term_normalizer

logger = logging.getLogger(__name__)


class MedicalMetadataExtractor(BaseExtractor):
    """Extracts medical metadata once per document and stamps it onto all nodes."""

    def __init__(
        self,
        llm_client: MetadataLLMClient | None = None,
        normalizer: TermNormalizer | None = None,
        **kwargs,
    ):
        """
        Initialize the LlamaIndex metadata extractor.

        Args:
            llm_client: Optional custom LLM client for extraction
            normalizer: Optional custom term normalizer
            **kwargs: Additional arguments for BaseExtractor
        """
        super().__init__(**kwargs)
        # Store as private attributes to avoid Pydantic validation
        object.__setattr__(self, "_llm_client", llm_client or get_metadata_llm_client())
        object.__setattr__(self, "_normalizer", normalizer or get_term_normalizer())
        object.__setattr__(self, "_document_cache", {})

    def _extract_relevant_text(self, nodes: Sequence[BaseNode]) -> str:
        """
        Extract relevant text for metadata extraction from LlamaIndex nodes.

        Prioritizes:
        1. Abstract sections
        2. Methods sections
        3. First few nodes

        Args:
            nodes: List of LlamaIndex nodes

        Returns:
            Combined text for metadata extraction
        """
        parts = []
        abstract_found = False
        methods_found = False

        for node in nodes:
            if not isinstance(node, TextNode):
                continue

            content = node.get_content()
            section_type = node.metadata.get("section_type", "").lower()
            chunk_type = node.metadata.get("chunk_type", "").lower()

            # Collect abstract
            if not abstract_found and ("abstract" in section_type or chunk_type == "abstract"):
                parts.append(f"ABSTRACT:\n{content[:2000]}")
                abstract_found = True

            # Collect methods
            elif not methods_found and "method" in section_type:
                parts.append(f"METHODS:\n{content[:2000]}")
                methods_found = True

            # Stop early if we have enough context
            if abstract_found and methods_found:
                break

            # Limit parts for token efficiency
            if len(parts) < 3:
                parts.append(content[:1500])

        # Combine all parts
        combined_text = "\n\n".join(parts)

        # If still empty, use first node content
        if not combined_text.strip() and nodes:
            combined_text = nodes[0].get_content()[:3000]

        return combined_text


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
                    # Extract relevant text from nodes
                    text = self._extract_relevant_text(nodes)

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
                        }
                    else:
                        # Get document title from first node if available
                        doc_title = nodes[0].metadata.get("title", "research paper") if nodes else "research paper"

                        # Extract medical terms using LLM
                        raw_metadata = await self._llm_client.extract_metadata(
                            text=text,
                            source_description=doc_title,
                        )

                        # Normalize terms to dropdown values
                        normalized_metadata = self._normalizer.normalize(raw_metadata)

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
