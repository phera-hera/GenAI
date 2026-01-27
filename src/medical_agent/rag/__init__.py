"""
Medical RAG using LlamaIndex CitationQueryEngine.

Provides direct retrieval with inline citations from medical research papers.

Usage:
    from medical_agent.rag import query_medical_rag

    response, citations = await query_medical_rag(
        ph_value=4.8,
        health_profile={"age": 28, "symptoms": ["mild discharge"]},
    )
    print(response.agent_reply)
"""

from medical_agent.rag.llamaindex_retrieval import (
    MedicalAnalysisResponse,
    build_citation_query_engine,
    query_medical_rag,
)

__all__ = [
    "build_citation_query_engine",
    "query_medical_rag",
    "MedicalAnalysisResponse",
]

