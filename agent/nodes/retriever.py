"""
Retriever Node

Retrieves relevant research paper chunks from the vector store:
- Uses multiple query variations for better recall
- Filters and reranks results
- Extracts citations from retrieved chunks
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.schema import QueryBundle

from agent.state import AgentState, RetrievedChunk
from app.core.config import settings
from app.services.langfuse_client import get_langfuse_client
from rag.retriever import (
    MedicalPaperRetriever,
    MultiQueryRetriever,
    RerankStrategy,
    RetrievalConfig,
    create_retriever,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def convert_node_to_chunk(node) -> dict[str, Any]:
    """
    Convert a LlamaIndex NodeWithScore to a RetrievedChunk dict.

    Args:
        node: NodeWithScore from the retriever

    Returns:
        Dict representation of the chunk
    """
    metadata = node.node.metadata or {}

    return RetrievedChunk(
        chunk_id=str(node.node.id_),
        paper_id=metadata.get("paper_id", ""),
        content=node.node.get_content(),
        chunk_type=metadata.get("chunk_type", "other"),
        score=node.score or 0.0,
        paper_title=metadata.get("paper_title"),
        paper_authors=metadata.get("paper_authors"),
        paper_doi=metadata.get("paper_doi"),
        section_title=metadata.get("section_title"),
        page_number=metadata.get("page_number"),
        metadata={
            k: v
            for k, v in metadata.items()
            if k
            not in [
                "paper_id",
                "paper_title",
                "paper_authors",
                "paper_doi",
                "section_title",
                "page_number",
                "chunk_type",
            ]
        },
    ).to_dict()


async def retrieve_with_queries(
    queries: list[str],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve chunks using multiple queries and merge results.

    Args:
        queries: List of search queries
        top_k: Maximum number of results per query

    Returns:
        List of unique retrieved chunks, sorted by score
    """
    # Configure retriever with reranking
    config = RetrievalConfig(
        top_k=top_k,
        rerank_strategy=RerankStrategy.CHUNK_TYPE_BOOST,
        include_paper_metadata=True,
        include_citations=True,
    )

    retriever = MedicalPaperRetriever(config=config)

    # Collect results from all queries
    all_results: dict[str, dict[str, Any]] = {}

    for query in queries:
        try:
            query_bundle = QueryBundle(query_str=query)
            nodes = await retriever._aretrieve(query_bundle)

            for node in nodes:
                chunk_dict = convert_node_to_chunk(node)
                chunk_id = chunk_dict["chunk_id"]

                # Keep the highest scoring occurrence
                if chunk_id not in all_results or all_results[chunk_id]["score"] < chunk_dict["score"]:
                    all_results[chunk_id] = chunk_dict

        except Exception as e:
            logger.warning(f"Retrieval failed for query '{query[:50]}...': {e}")
            continue

    # Sort by score and return top results
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return sorted_results[:top_k]


def retriever_node(state: AgentState) -> AgentState:
    """
    Synchronous wrapper for the retriever node.

    For LangGraph compatibility, this is the main entry point.
    """
    import asyncio

    return asyncio.run(aretriever_node(state))


async def aretriever_node(state: AgentState) -> AgentState:
    """
    Retriever Node - Retrieve relevant research chunks.

    This node:
    1. Uses search queries from query analysis
    2. Retrieves from vector store with reranking
    3. Deduplicates and merges results
    4. Logs retrieval to Langfuse

    Args:
        state: Current agent state

    Returns:
        Updated agent state with retrieved_chunks
    """
    logger.info("Retriever: Starting document retrieval")

    # Get trace for Langfuse
    trace = None
    langfuse = get_langfuse_client()
    if langfuse.is_configured() and state.get("langfuse_trace_id"):
        trace = langfuse.client.trace(id=state["langfuse_trace_id"])

    try:
        # Get search queries from query analysis
        queries = state.get("retrieval_query_variations", [])

        if not queries:
            # Fall back to default query based on pH
            query_analysis = state.get("query_analysis", {})
            ph_value = query_analysis.get("ph_value", state.get("ph_value", 4.5))
            queries = [f"vaginal pH {ph_value} health implications research"]

        # Retrieve chunks
        retrieved_chunks = await retrieve_with_queries(
            queries=queries,
            top_k=settings.vector_similarity_top_k,
        )

        # Log to Langfuse
        if trace:
            langfuse.log_retrieval(
                trace=trace,
                name="vector_retrieval",
                query="; ".join(queries[:3]),  # First 3 queries
                documents=[
                    {
                        "chunk_id": c["chunk_id"],
                        "paper_title": c.get("paper_title"),
                        "score": c["score"],
                        "chunk_type": c["chunk_type"],
                    }
                    for c in retrieved_chunks[:5]  # Top 5 for logging
                ],
                metadata={
                    "num_queries": len(queries),
                    "num_results": len(retrieved_chunks),
                },
            )

        state["retrieved_chunks"] = retrieved_chunks

        logger.info(f"Retriever complete: {len(retrieved_chunks)} chunks retrieved")

    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        state["errors"].append(f"Retrieval error: {str(e)}")
        state["retrieved_chunks"] = []

    return state


def format_chunks_for_context(chunks: list[dict[str, Any]], max_chunks: int = 10) -> str:
    """
    Format retrieved chunks as context for LLM.

    Args:
        chunks: List of retrieved chunk dicts
        max_chunks: Maximum number of chunks to include

    Returns:
        Formatted string with chunk content and metadata
    """
    if not chunks:
        return "No relevant research papers found."

    lines = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        paper_info = []
        if chunk.get("paper_title"):
            paper_info.append(f"Title: {chunk['paper_title']}")
        if chunk.get("paper_authors"):
            paper_info.append(f"Authors: {chunk['paper_authors']}")
        if chunk.get("section_title"):
            paper_info.append(f"Section: {chunk['section_title']}")

        header = f"[Source {i}]"
        if paper_info:
            header += f" ({'; '.join(paper_info)})"

        lines.append(header)
        lines.append(chunk.get("content", ""))
        lines.append("")  # Blank line between chunks

    return "\n".join(lines)


def extract_citations_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract unique citations from retrieved chunks.

    Args:
        chunks: List of retrieved chunk dicts

    Returns:
        List of unique citation dicts
    """
    seen_papers = set()
    citations = []

    for chunk in chunks:
        paper_id = chunk.get("paper_id")
        if paper_id and paper_id not in seen_papers:
            seen_papers.add(paper_id)
            citations.append({
                "paper_id": paper_id,
                "title": chunk.get("paper_title", "Unknown"),
                "authors": chunk.get("paper_authors", "Unknown"),
                "doi": chunk.get("paper_doi"),
                "score": chunk.get("score", 0.0),
            })

    return citations


