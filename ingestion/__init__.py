"""
Ingestion Pipeline for Processing Medical Research Papers

This module provides the complete pipeline for:
1. Parsing PDFs using Azure Document Intelligence (with PyMuPDF fallback)
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
    DocumentSection,
    ExtractedTable,
    FallbackPDFParser,
    MedicalPDFParser,
    PaperMetadata,
    ParsedDocument,
    PDFParserFacade,
    SectionType,
    TableCell,
    get_fallback_parser,
    get_parser_facade,
    get_pdf_parser,
    parse_pdf,
)

# Chunkers
from .chunkers import ChunkedSection, SectionChunker

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
    "MedicalPDFParser",
    "FallbackPDFParser",
    # Factory functions
    "get_parser_facade",
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
    "ChunkedSection",
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
