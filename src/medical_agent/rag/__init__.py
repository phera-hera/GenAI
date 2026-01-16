"""
RAG Module - LlamaIndex integration for medical paper retrieval.

This module provides:
- Hybrid search combining semantic (vector) and BM25 (keyword) retrieval
- Reciprocal Rank Fusion (RRF) for combining search results
- Vector index setup with pgvector
- Custom retriever with filtering and reranking
- Query engine for medical RAG queries
- pH analysis engine for specialized health queries

Usage:
    from rag import create_query_engine, create_hybrid_retriever

    # Create a hybrid retriever (default)
    retriever = create_hybrid_retriever(top_k=10, semantic_weight=0.7, bm25_weight=0.3)

    # Create a query engine with hybrid search
    engine = create_query_engine(top_k=10)
    result = await engine.aquery("What causes vaginal pH imbalance?")

    # Create a pH analyzer
    analyzer = create_ph_analyzer()
    result = await analyzer.analyze_ph(ph_value=5.2, symptoms=["itching"])
"""

from medical_agent.rag.index import (
    MedicalRAGIndex,
    VectorStoreConfig,
    create_node_from_chunk,
    create_node_with_score,
    get_medical_rag_index,
)
from medical_agent.rag.query_engine import (
    MedicalQueryEngine,
    PHAnalysisEngine,
    QueryResult,
    create_ph_analyzer,
    create_query_engine,
)
from medical_agent.rag.retriever import (
    MedicalPaperRetriever,
    MultiQueryRetriever,
    RerankStrategy,
    RetrievalConfig,
    RetrievalResult,
    SearchMode,
    create_bm25_retriever,
    create_hybrid_retriever,
    create_multi_query_retriever,
    create_retriever,
    create_semantic_retriever,
)

__all__ = [
    # Index
    "MedicalRAGIndex",
    "VectorStoreConfig",
    "get_medical_rag_index",
    "create_node_from_chunk",
    "create_node_with_score",
    # Retriever
    "MedicalPaperRetriever",
    "MultiQueryRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "RerankStrategy",
    "SearchMode",
    "create_retriever",
    "create_semantic_retriever",
    "create_bm25_retriever",
    "create_hybrid_retriever",
    "create_multi_query_retriever",
    # Query Engine
    "MedicalQueryEngine",
    "PHAnalysisEngine",
    "QueryResult",
    "create_query_engine",
    "create_ph_analyzer",
]
