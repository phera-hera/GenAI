"""
Ingestion Pipeline for Processing Medical Research Papers

This module provides the complete pipeline for:
1. Parsing PDFs using LlamaParser (with PyMuPDF fallback)
2. Chunking papers into semantic sections
3. Generating embeddings using Azure OpenAI
4. Storing chunks in pgvector for retrieval

Quick start:
    from ingestion import process_pdf, search_similar
    
    # Process a PDF
    with open("paper.pdf", "rb") as f:
        result = await process_pdf(f.read(), gcp_path="papers/paper.pdf")
    
    print(result.summary())
    
    # Search for similar content
    results = await search_similar("vaginal pH levels and bacterial infections")
    for r in results:
        print(f"Score: {r.score:.3f} - {r.content[:100]}...")
"""

# Parsers
from .parsers import (
    DoclingPDFParser,
    DocumentSection,
    ExtractedTable,
    FallbackPDFParser,
    MedicalPDFParser,
    PaperMetadata,
    ParsedDocument,
    PDFParserFacade,
    SectionType,
    TableCell,
    get_docling_parser,
    get_fallback_parser,
    get_parser_facade,
    get_pdf_parser,
    parse_pdf,
)

# Chunkers
from .chunkers import (
    ChunkedSection,
    DoclingHierarchicalChunker,
    SectionChunker,
    get_docling_chunker,
)

# Metadata extraction
from .metadata import (
    ExtractedMetadata,
    MedicalMetadataExtractor,
    MetadataLLMClient,
    TableMetadata,
    TermNormalizer,
    get_metadata_extractor,
    get_metadata_llm_client,
    get_term_normalizer,
)

# Embedders
from .embedders import (
    AsyncAzureEmbedder,
    AzureEmbedder,
    EmbeddedChunk,
    EmbeddingResult,
    get_async_embedder,
    get_embedder,
)

# Storage
from .storage import (
    DistanceMetric,
    SearchQuery,
    SearchResult,
    StorageResult,
    VectorStore,
    get_vector_store,
)

# Pipeline
from .pipeline import (
    IngestionPipeline,
    PipelineConfig,
    PipelineResult,
    process_pdf,
    search_similar,
)

__all__ = [
    # Convenience functions
    "parse_pdf",
    "process_pdf",
    "search_similar",
    # Parser classes
    "PDFParserFacade",
    "DoclingPDFParser",
    "MedicalPDFParser",
    "FallbackPDFParser",
    # Factory functions
    "get_parser_facade",
    "get_docling_parser",
    "get_pdf_parser",
    "get_fallback_parser",
    "get_embedder",
    "get_async_embedder",
    "get_vector_store",
    # Data classes
    "ParsedDocument",
    "DocumentSection",
    "ExtractedTable",
    "TableCell",
    "PaperMetadata",
    "SectionType",
    # Chunking
    "SectionChunker",
    "DoclingHierarchicalChunker",
    "get_docling_chunker",
    "ChunkedSection",
    # Metadata extraction
    "MedicalMetadataExtractor",
    "MetadataLLMClient",
    "TermNormalizer",
    "ExtractedMetadata",
    "TableMetadata",
    "get_metadata_extractor",
    "get_metadata_llm_client",
    "get_term_normalizer",
    # Embedding
    "AzureEmbedder",
    "AsyncAzureEmbedder",
    "EmbeddedChunk",
    "EmbeddingResult",
    # Storage
    "VectorStore",
    "SearchQuery",
    "SearchResult",
    "StorageResult",
    "DistanceMetric",
    # Pipeline
    "IngestionPipeline",
    "PipelineConfig",
    "PipelineResult",
]
