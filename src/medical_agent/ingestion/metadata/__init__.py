"""
Metadata Extraction for Medical Research Papers

Provides LLM-based metadata extraction that runs after parsing and before chunking.
Extracts medical context tags (ethnicity, diagnoses, symptoms) that are stamped onto every chunk.

Main components:
- MetadataLLMClient: GPT-4o based extraction with structured output
- TermNormalizer: Maps extracted terms to dropdown values
- MedicalMetadataExtractor: Orchestrates extraction and normalization
- ExtractedMetadata: Structured metadata result

Usage:
    from ingestion.metadata import MedicalMetadataExtractor

    extractor = MedicalMetadataExtractor()
    metadata = await extractor.extract(parsed_document)
"""

from .extractor import MedicalMetadataExtractor, get_metadata_extractor
from .llm_client import MetadataLLMClient, get_metadata_llm_client
from .normalizer import TermNormalizer, get_term_normalizer
from .types import ExtractedMetadata, TableMetadata

__all__ = [
    # Main extractor
    "MedicalMetadataExtractor",
    "get_metadata_extractor",
    # LLM client
    "MetadataLLMClient",
    "get_metadata_llm_client",
    # Normalizer
    "TermNormalizer",
    "get_term_normalizer",
    # Data types
    "ExtractedMetadata",
    "TableMetadata",
]
