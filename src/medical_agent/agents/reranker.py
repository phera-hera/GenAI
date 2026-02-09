"""Cross-encoder reranker for retrieved nodes."""

import logging
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Module-level singleton (loaded once, ~200MB model)
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Get or create the cross-encoder reranker (singleton)."""
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder model: ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Cross-encoder loaded")
    return _reranker


def rerank_nodes(
    query: str,
    nodes: list[NodeWithScore],
    top_k: int = 5,
) -> list[NodeWithScore]:
    """
    Rerank retrieved nodes using a cross-encoder.

    Args:
        query: Original user query (not enhanced)
        nodes: Retrieved nodes from hybrid search
        top_k: Number of top results to keep after reranking

    Returns:
        Top-k nodes reranked by cross-encoder score
    """
    if not nodes:
        return nodes

    reranker = get_reranker()

    # Create query-document pairs for cross-encoder
    pairs = [(query, node.node.text) for node in nodes]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach reranker scores and sort
    for node, score in zip(nodes, scores):
        node.score = float(score)  # Replace hybrid score with reranker score

    # Sort by reranker score (descending) and keep top_k
    reranked = sorted(nodes, key=lambda n: n.score, reverse=True)[:top_k]

    logger.info(
        f"Reranked {len(nodes)} → {len(reranked)} nodes. "
        f"Top score: {reranked[0].score:.3f}, Bottom: {reranked[-1].score:.3f}"
    )

    return reranked
