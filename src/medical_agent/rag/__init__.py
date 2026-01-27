"""
Medical RAG using LlamaIndex retriever.retrieve().

Provides structured node retrieval from medical research papers.
Returns NodeWithScore objects for use in LangGraph workflows.

Usage:
    from medical_agent.rag import retrieve_nodes

    # Retrieve nodes for a query
    nodes = retrieve_nodes("pH 5.2 with discharge")

    for node in nodes:
        print(node.node.text)
        print(node.node.metadata["paper_id"])
"""

from medical_agent.rag.llamaindex_retrieval import (
    build_retriever,
    retrieve_nodes,
)

__all__ = [
    "build_retriever",
    "retrieve_nodes",
]

