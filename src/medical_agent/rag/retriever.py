"""
Custom Medical Paper Retriever with Hybrid Search

Provides a specialized retriever for medical paper chunks that supports:
- Hybrid search combining semantic (vector) and BM25 (keyword) retrieval
- Vector similarity search via pgvector
- BM25 full-text search via PostgreSQL tsvector
- Reciprocal Rank Fusion (RRF) for combining results
- Filtering by chunk type, paper, and metadata
- Reranking strategies
- Citation extraction
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from medical_agent.core.config import settings
from medical_agent.core.exceptions import DatabaseException
from medical_agent.infrastructure.database.models import ChunkType
from medical_agent.infrastructure.database.session import get_session_context
from medical_agent.ingestion.embedders.azure_embedder import get_async_embedder
from medical_agent.ingestion.storage.types import SearchQuery, SearchResult
from medical_agent.ingestion.storage.vector_store import MedicalPGVectorStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search modes for retrieval."""

    SEMANTIC = "semantic"  # Vector similarity only
    BM25 = "bm25"  # Keyword/full-text only
    HYBRID = "hybrid"  # Combine semantic + BM25


class RerankStrategy(str, Enum):
    """Available reranking strategies for search results."""

    NONE = "none"
    SCORE_DECAY = "score_decay"  # Apply decay based on rank
    CHUNK_TYPE_BOOST = "chunk_type_boost"  # Boost certain chunk types
    RECENCY_BOOST = "recency_boost"  # Boost recent papers
    RRF = "rrf"  # Reciprocal Rank Fusion (for hybrid search)


@dataclass
class RetrievalConfig:
    """Configuration for the medical paper retriever with hybrid search support."""

    # Basic search settings
    top_k: int = 10
    similarity_threshold: float | None = None

    # Search mode
    search_mode: SearchMode = SearchMode.HYBRID  # Default to hybrid search

    # Hybrid search settings
    semantic_weight: float = 0.7  # Weight for semantic results (0-1)
    bm25_weight: float = 0.3  # Weight for BM25 results (0-1)
    rrf_k: int = 60  # RRF constant (higher = more weight to lower ranks)

    # Filtering
    chunk_types: list[ChunkType] | None = None
    paper_ids: list[uuid.UUID] | None = None
    exclude_paper_ids: list[uuid.UUID] | None = None

    # Medical metadata filtering (NEW!)
    filter_ethnicities: list[str] | None = None
    filter_diagnoses: list[str] | None = None
    filter_symptoms: list[str] | None = None
    filter_menstrual_status: list[str] | None = None
    filter_birth_control: list[str] | None = None
    filter_hormone_therapy: list[str] | None = None
    filter_fertility_treatments: list[str] | None = None

    # Reranking (applied after fusion)
    rerank_strategy: RerankStrategy = RerankStrategy.NONE
    chunk_type_boosts: dict[ChunkType, float] | None = None

    # Metadata inclusion
    include_paper_metadata: bool = True
    include_citations: bool = True

    # Advanced options
    fetch_k: int | None = None  # Over-fetch for reranking (default: top_k * 2)

    def __post_init__(self) -> None:
        """Set computed defaults and validate."""
        if self.fetch_k is None:
            self.fetch_k = self.top_k * 2

        # Normalize weights
        total_weight = self.semantic_weight + self.bm25_weight
        if total_weight > 0:
            self.semantic_weight /= total_weight
            self.bm25_weight /= total_weight


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    nodes: list[NodeWithScore]
    query_text: str
    total_candidates: int = 0
    processing_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        return len(self.nodes) > 0

    @property
    def top_node(self) -> NodeWithScore | None:
        return self.nodes[0] if self.nodes else None

    def get_citations(self) -> list[dict[str, Any]]:
        """Extract citation information from results."""
        citations = []
        seen_papers = set()

        for node in self.nodes:
            paper_id = node.metadata.get("paper_id")
            if paper_id and paper_id not in seen_papers:
                seen_papers.add(paper_id)
                citations.append({
                    "paper_id": paper_id,
                    "paper_title": node.metadata.get("paper_title"),
                    "paper_authors": node.metadata.get("paper_authors"),
                    "paper_doi": node.metadata.get("paper_doi"),
                    "score": node.score,
                })

        return citations


class MedicalPaperRetriever(BaseRetriever):
    """
    Custom retriever for medical paper chunks with hybrid search support.

    Wraps the pgvector-based MedicalPGVectorStore and provides LlamaIndex-compatible
    retrieval with hybrid search (semantic + BM25) and advanced features.

    This retriever:
    - Supports hybrid search combining semantic and BM25 results
    - Uses Reciprocal Rank Fusion (RRF) to merge results
    - Uses async database operations for efficiency
    - Supports filtering by chunk type (abstract, results, etc.)
    - Provides reranking strategies for better relevance
    - Extracts citation metadata from retrieved chunks

    Args:
        config: RetrievalConfig with search settings
        vector_store: Optional MedicalPGVectorStore instance
        session: Optional AsyncSession for database operations
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        vector_store: MedicalPGVectorStore | None = None,
        session: AsyncSession | None = None,
    ):
        super().__init__()
        self.config = config or RetrievalConfig()
        self._vector_store = vector_store
        self._session = session
        self._embedder = get_async_embedder()

    @property
    def vector_store(self) -> MedicalPGVectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            self._vector_store = MedicalPGVectorStore()
        return self._vector_store

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """
        Synchronous retrieve - raises error, use async version.

        LlamaIndex BaseRetriever requires this method, but we use async.
        """
        raise NotImplementedError(
            "Use async_retrieve() for this retriever. "
            "MedicalPaperRetriever requires async database operations."
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """
        Async retrieve implementation with hybrid search support.

        Args:
            query_bundle: Query with text and optional embedding

        Returns:
            List of NodeWithScore sorted by relevance
        """
        import time

        start_time = time.time()
        query_text = query_bundle.query_str

        logger.debug(
            f"Retrieving for query: {query_text[:100]}... "
            f"(mode={self.config.search_mode.value})"
        )

        try:
            # Execute search based on mode
            if self.config.search_mode == SearchMode.HYBRID:
                results = await self._hybrid_search(query_bundle)
            elif self.config.search_mode == SearchMode.BM25:
                results = await self._bm25_search(query_text)
            else:  # SEMANTIC
                results = await self._semantic_search(query_bundle)

            # Apply exclusions
            if self.config.exclude_paper_ids:
                exclude_set = set(self.config.exclude_paper_ids)
                results = [r for r in results if r.paper_id not in exclude_set]

            # Apply reranking (after fusion)
            results = self._apply_reranking(results)

            # Limit to top_k
            results = results[: self.config.top_k]

            # Convert to LlamaIndex nodes
            nodes = [self._result_to_node(r) for r in results]

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Retrieved {len(nodes)} chunks in {elapsed_ms}ms "
                f"(mode={self.config.search_mode.value}, query: {query_text[:50]}...)"
            )

            return nodes

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise DatabaseException(f"Retrieval failed: {e}")

    async def _semantic_search(
        self, query_bundle: QueryBundle
    ) -> list[SearchResult]:
        """Perform semantic (vector) search only."""
        # Get query embedding
        if query_bundle.embedding is not None:
            query_embedding = query_bundle.embedding
        else:
            query_embedding = await self._embedder.embed_single(query_bundle.query_str)

        # Build search query
        search_query = SearchQuery(
            embedding=query_embedding,
            top_k=self.config.fetch_k or self.config.top_k * 2,
            distance_metric="cosine",
            paper_ids=self.config.paper_ids,
            chunk_types=self.config.chunk_types,
            min_score=self.config.similarity_threshold,
            include_paper_metadata=self.config.include_paper_metadata,
            # Medical metadata filters (NEW!)
            filter_ethnicities=self.config.filter_ethnicities,
            filter_diagnoses=self.config.filter_diagnoses,
            filter_symptoms=self.config.filter_symptoms,
            filter_menstrual_status=self.config.filter_menstrual_status,
            filter_birth_control=self.config.filter_birth_control,
            filter_hormone_therapy=self.config.filter_hormone_therapy,
            filter_fertility_treatments=self.config.filter_fertility_treatments,
        )

        # Execute search
        if self._session:
            return await self.vector_store.similarity_search(
                self._session, search_query
            )
        else:
            async with get_session_context() as session:
                return await self.vector_store.similarity_search(
                    session, search_query
                )

    async def _bm25_search(self, query_text: str) -> list[SearchResult]:
        """Perform BM25 (full-text) search only."""
        if self._session:
            return await self.vector_store.bm25_search(
                self._session,
                query_text=query_text,
                top_k=self.config.fetch_k or self.config.top_k * 2,
                paper_ids=self.config.paper_ids,
                chunk_types=self.config.chunk_types,
                include_paper_metadata=self.config.include_paper_metadata,
                # Medical metadata filters (NEW!)
                filter_ethnicities=self.config.filter_ethnicities,
                filter_diagnoses=self.config.filter_diagnoses,
                filter_symptoms=self.config.filter_symptoms,
                filter_menstrual_status=self.config.filter_menstrual_status,
                filter_birth_control=self.config.filter_birth_control,
                filter_hormone_therapy=self.config.filter_hormone_therapy,
                filter_fertility_treatments=self.config.filter_fertility_treatments,
            )
        else:
            async with get_session_context() as session:
                return await self.vector_store.bm25_search(
                    session,
                    query_text=query_text,
                    top_k=self.config.fetch_k or self.config.top_k * 2,
                    paper_ids=self.config.paper_ids,
                    chunk_types=self.config.chunk_types,
                    include_paper_metadata=self.config.include_paper_metadata,
                    # Medical metadata filters (NEW!)
                    filter_ethnicities=self.config.filter_ethnicities,
                    filter_diagnoses=self.config.filter_diagnoses,
                    filter_symptoms=self.config.filter_symptoms,
                    filter_menstrual_status=self.config.filter_menstrual_status,
                    filter_birth_control=self.config.filter_birth_control,
                    filter_hormone_therapy=self.config.filter_hormone_therapy,
                    filter_fertility_treatments=self.config.filter_fertility_treatments,
                )

    async def _hybrid_search(
        self, query_bundle: QueryBundle
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining semantic and BM25.

        Uses Reciprocal Rank Fusion (RRF) to combine results.
        """
        query_text = query_bundle.query_str

        # Run searches sequentially to avoid concurrent session usage
        if self._session:
            semantic_results = await self._semantic_search(query_bundle)
            bm25_results = await self._bm25_search(query_text)
        else:
            async with get_session_context() as session:
                self._session = session
                semantic_results = await self._semantic_search(query_bundle)
                bm25_results = await self._bm25_search(query_text)
                self._session = None

        logger.debug(
            f"Hybrid search: {len(semantic_results)} semantic, "
            f"{len(bm25_results)} BM25 results"
        )

        # Combine using RRF
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            semantic_weight=self.config.semantic_weight,
            bm25_weight=self.config.bm25_weight,
            k=self.config.rrf_k,
        )

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[SearchResult],
        bm25_results: list[SearchResult],
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(weight / (k + rank)) for each result list

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            semantic_weight: Weight for semantic scores
            bm25_weight: Weight for BM25 scores
            k: RRF constant (higher = more weight to lower ranks)

        Returns:
            Fused and sorted results
        """
        # Build score maps
        rrf_scores: dict[uuid.UUID, float] = {}
        result_map: dict[uuid.UUID, SearchResult] = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += semantic_weight / (k + rank)
            result_map[chunk_id] = result

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += bm25_weight / (k + rank)
            # Update result_map (prefer semantic result if exists)
            if chunk_id not in result_map:
                result_map[chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results with RRF score
        fused_results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            # Update score to RRF score (normalized)
            result.score = rrf_scores[chunk_id]
            fused_results.append(result)

        logger.debug(
            f"RRF fusion: {len(fused_results)} unique results from "
            f"{len(semantic_results)} semantic + {len(bm25_results)} BM25"
        )

        return fused_results

    def _apply_reranking(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Apply configured reranking strategy.

        Args:
            results: Initial search results

        Returns:
            Reranked results
        """
        if self.config.rerank_strategy == RerankStrategy.NONE:
            return results

        if self.config.rerank_strategy == RerankStrategy.SCORE_DECAY:
            return self._rerank_score_decay(results)

        if self.config.rerank_strategy == RerankStrategy.CHUNK_TYPE_BOOST:
            return self._rerank_chunk_type_boost(results)

        return results

    def _rerank_score_decay(
        self,
        results: list[SearchResult],
        decay_factor: float = 0.95,
    ) -> list[SearchResult]:
        """
        Apply score decay based on original rank.

        Results keep their order but scores are adjusted to create
        more separation between highly ranked and lower ranked items.

        Args:
            results: Search results
            decay_factor: Multiplier applied per rank position

        Returns:
            Results with adjusted scores
        """
        for i, result in enumerate(results):
            # Apply exponential decay
            result.score = result.score * (decay_factor**i)

        return results

    def _rerank_chunk_type_boost(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Boost scores based on chunk type relevance.

        For medical queries, abstracts and results sections are often
        more valuable than methods or references.

        Args:
            results: Search results

        Returns:
            Re-sorted results with boosted scores
        """
        # Default boosts if not configured
        boosts = self.config.chunk_type_boosts or {
            ChunkType.ABSTRACT: 1.2,
            ChunkType.RESULTS: 1.15,
            ChunkType.CONCLUSION: 1.1,
            ChunkType.DISCUSSION: 1.05,
            ChunkType.INTRODUCTION: 1.0,
            ChunkType.METHODS: 0.9,
            ChunkType.TABLE: 1.0,
            ChunkType.REFERENCES: 0.7,
            ChunkType.OTHER: 0.8,
        }

        for result in results:
            try:
                chunk_type = ChunkType(result.chunk_type)
                boost = boosts.get(chunk_type, 1.0)
                result.score = result.score * boost
            except (ValueError, KeyError):
                pass  # Keep original score if chunk type not recognized

        # Re-sort by boosted score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _result_to_node(self, result: SearchResult) -> NodeWithScore:
        """
        Convert a SearchResult to a LlamaIndex NodeWithScore.

        Args:
            result: Vector store search result

        Returns:
            NodeWithScore for LlamaIndex compatibility
        """
        # Build comprehensive metadata
        metadata = {
            "chunk_id": str(result.chunk_id),
            "paper_id": str(result.paper_id),
            "chunk_type": result.chunk_type,
            "section_title": result.section_title,
            "page_number": result.page_number,
            **(result.chunk_metadata or {}),
        }

        # Add paper metadata if available
        if result.paper_title:
            metadata["paper_title"] = result.paper_title
        if result.paper_authors:
            metadata["paper_authors"] = result.paper_authors
        if result.paper_doi:
            metadata["paper_doi"] = result.paper_doi

        # Add citation string
        metadata["citation"] = result.citation

        # Create text node
        node = TextNode(
            id_=str(result.chunk_id),
            text=result.content,
            metadata=metadata,
        )

        return NodeWithScore(node=node, score=result.score)


class MultiQueryRetriever:
    """
    Retriever that generates multiple query variations for better recall.

    Uses the LLM to generate alternative phrasings of the query,
    retrieves for each, and merges/deduplicates results.

    This is useful for medical queries where terminology varies
    (e.g., "pH imbalance" vs "vaginal acidosis" vs "elevated pH levels").
    """

    def __init__(
        self,
        base_retriever: MedicalPaperRetriever,
        llm=None,
        num_queries: int = 3,
    ):
        self.base_retriever = base_retriever
        self._llm = llm
        self.num_queries = num_queries

    @property
    def llm(self):
        """Get or create the LLM."""
        if self._llm is None:
            from medical_agent.infrastructure.azure_openai import get_llama_index_llm

            self._llm = get_llama_index_llm()
        return self._llm

    async def generate_query_variations(self, query: str) -> list[str]:
        """
        Generate alternative phrasings of the query.

        Args:
            query: Original user query

        Returns:
            List of query variations including the original
        """
        prompt = f"""Generate {self.num_queries - 1} alternative phrasings of this medical query.
Focus on different medical terminology that might appear in research papers.
Return only the queries, one per line.

Original query: {query}

Alternative queries:"""

        try:
            response = await self.llm.acomplete(prompt)
            variations = [query]  # Include original
            for line in response.text.strip().split("\n"):
                line = line.strip()
                if line and line != query:
                    # Remove numbering if present
                    if line[0].isdigit() and "." in line[:3]:
                        line = line.split(".", 1)[1].strip()
                    variations.append(line)

            return variations[: self.num_queries]

        except Exception as e:
            logger.warning(f"Failed to generate query variations: {e}")
            return [query]

    async def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve using multiple query variations.

        Args:
            query: Original user query

        Returns:
            Merged and deduplicated results
        """
        import time

        start_time = time.time()

        # Generate variations
        variations = await self.generate_query_variations(query)
        logger.debug(f"Generated {len(variations)} query variations")

        # Retrieve for each variation
        all_nodes: dict[str, NodeWithScore] = {}
        total_candidates = 0

        for variation in variations:
            query_bundle = QueryBundle(query_str=variation)
            nodes = await self.base_retriever._aretrieve(query_bundle)
            total_candidates += len(nodes)

            # Merge, keeping highest score per node
            for node in nodes:
                node_id = node.node.id_
                if node_id not in all_nodes or all_nodes[node_id].score < node.score:
                    all_nodes[node_id] = node

        # Sort by score and limit
        sorted_nodes = sorted(all_nodes.values(), key=lambda n: n.score, reverse=True)
        final_nodes = sorted_nodes[: self.base_retriever.config.top_k]

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RetrievalResult(
            nodes=final_nodes,
            query_text=query,
            total_candidates=total_candidates,
            processing_time_ms=elapsed_ms,
            metadata={
                "query_variations": variations,
                "unique_results": len(all_nodes),
            },
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_retriever(
    top_k: int | None = None,
    chunk_types: list[ChunkType] | None = None,
    rerank: bool = True,
    search_mode: SearchMode = SearchMode.HYBRID,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
    **kwargs: Any,
) -> MedicalPaperRetriever:
    """
    Create a configured MedicalPaperRetriever with hybrid search.

    Args:
        top_k: Number of results to return
        chunk_types: Filter by chunk types
        rerank: Whether to apply reranking
        search_mode: Search mode (HYBRID, SEMANTIC, or BM25)
        semantic_weight: Weight for semantic results in hybrid mode (0-1)
        bm25_weight: Weight for BM25 results in hybrid mode (0-1)
        **kwargs: Additional RetrievalConfig parameters

    Returns:
        Configured retriever with hybrid search support
    """
    config = RetrievalConfig(
        top_k=top_k or settings.vector_similarity_top_k,
        chunk_types=chunk_types,
        search_mode=search_mode,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        rerank_strategy=(
            RerankStrategy.CHUNK_TYPE_BOOST if rerank else RerankStrategy.NONE
        ),
        **kwargs,
    )
    return MedicalPaperRetriever(config=config)


def create_semantic_retriever(
    top_k: int | None = None,
    **kwargs: Any,
) -> MedicalPaperRetriever:
    """
    Create a retriever using semantic (vector) search only.

    Args:
        top_k: Number of results to return
        **kwargs: Additional retriever configuration

    Returns:
        Configured semantic-only retriever
    """
    return create_retriever(
        top_k=top_k,
        search_mode=SearchMode.SEMANTIC,
        **kwargs,
    )


def create_bm25_retriever(
    top_k: int | None = None,
    **kwargs: Any,
) -> MedicalPaperRetriever:
    """
    Create a retriever using BM25 (keyword) search only.

    Args:
        top_k: Number of results to return
        **kwargs: Additional retriever configuration

    Returns:
        Configured BM25-only retriever
    """
    return create_retriever(
        top_k=top_k,
        search_mode=SearchMode.BM25,
        **kwargs,
    )


def create_hybrid_retriever(
    top_k: int | None = None,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
    **kwargs: Any,
) -> MedicalPaperRetriever:
    """
    Create a hybrid retriever combining semantic and BM25 search.

    Uses Reciprocal Rank Fusion (RRF) to combine results.

    Args:
        top_k: Number of results to return
        semantic_weight: Weight for semantic results (0-1)
        bm25_weight: Weight for BM25 results (0-1)
        **kwargs: Additional retriever configuration

    Returns:
        Configured hybrid retriever
    """
    return create_retriever(
        top_k=top_k,
        search_mode=SearchMode.HYBRID,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        **kwargs,
    )


def create_multi_query_retriever(
    top_k: int | None = None,
    num_queries: int = 3,
    search_mode: SearchMode = SearchMode.HYBRID,
    **kwargs: Any,
) -> MultiQueryRetriever:
    """
    Create a MultiQueryRetriever for improved recall with hybrid search.

    Args:
        top_k: Number of results to return
        num_queries: Number of query variations to generate
        search_mode: Search mode (HYBRID, SEMANTIC, or BM25)
        **kwargs: Additional retriever configuration

    Returns:
        Configured multi-query retriever with hybrid search
    """
    base_retriever = create_retriever(
        top_k=top_k, search_mode=search_mode, **kwargs
    )
    return MultiQueryRetriever(
        base_retriever=base_retriever,
        num_queries=num_queries,
    )

