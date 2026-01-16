"""
Tests for the RAG retriever module.

Tests cover:
- RetrievalConfig creation and defaults
- RetrievalResult functionality
- MedicalPaperRetriever configuration
- Hybrid search (semantic + BM25)
- Reciprocal Rank Fusion (RRF)
- Reranking strategies
- Factory functions
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from medical_agent.infrastructure.database.models import ChunkType
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


class TestRetrievalConfig:
    """Tests for RetrievalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetrievalConfig()

        assert config.top_k == 10
        assert config.similarity_threshold is None
        assert config.search_mode == SearchMode.HYBRID  # Default to hybrid
        assert config.chunk_types is None
        assert config.paper_ids is None
        assert config.rerank_strategy == RerankStrategy.NONE
        assert config.include_paper_metadata is True
        assert config.include_citations is True

    def test_default_hybrid_weights(self):
        """Test default hybrid search weights."""
        config = RetrievalConfig()

        # Weights should be normalized
        assert config.semantic_weight == pytest.approx(0.7)
        assert config.bm25_weight == pytest.approx(0.3)

    def test_hybrid_weight_normalization(self):
        """Test that hybrid weights are normalized."""
        config = RetrievalConfig(semantic_weight=2.0, bm25_weight=2.0)

        # Should be normalized to sum to 1.0
        assert config.semantic_weight == pytest.approx(0.5)
        assert config.bm25_weight == pytest.approx(0.5)

    def test_fetch_k_computed(self):
        """Test fetch_k is computed as top_k * 2."""
        config = RetrievalConfig(top_k=5)
        assert config.fetch_k == 10

    def test_fetch_k_override(self):
        """Test fetch_k can be explicitly set."""
        config = RetrievalConfig(top_k=5, fetch_k=15)
        assert config.fetch_k == 15

    def test_custom_configuration(self):
        """Test custom configuration values."""
        paper_ids = [uuid.uuid4(), uuid.uuid4()]
        chunk_types = [ChunkType.ABSTRACT, ChunkType.RESULTS]

        config = RetrievalConfig(
            top_k=20,
            similarity_threshold=0.7,
            search_mode=SearchMode.SEMANTIC,
            chunk_types=chunk_types,
            paper_ids=paper_ids,
            rerank_strategy=RerankStrategy.CHUNK_TYPE_BOOST,
        )

        assert config.top_k == 20
        assert config.similarity_threshold == 0.7
        assert config.search_mode == SearchMode.SEMANTIC
        assert config.chunk_types == chunk_types
        assert config.paper_ids == paper_ids
        assert config.rerank_strategy == RerankStrategy.CHUNK_TYPE_BOOST


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def _create_node(
        self,
        text: str = "Test content",
        score: float = 0.9,
        paper_id: str | None = None,
        paper_title: str | None = None,
    ) -> NodeWithScore:
        """Helper to create a NodeWithScore."""
        metadata = {}
        if paper_id:
            metadata["paper_id"] = paper_id
        if paper_title:
            metadata["paper_title"] = paper_title

        node = TextNode(
            id_=str(uuid.uuid4()),
            text=text,
            metadata=metadata,
        )
        return NodeWithScore(node=node, score=score)

    def test_empty_result(self):
        """Test empty result properties."""
        result = RetrievalResult(nodes=[], query_text="test")

        assert result.has_results is False
        assert result.top_node is None
        assert result.get_citations() == []

    def test_result_with_nodes(self):
        """Test result with nodes."""
        nodes = [
            self._create_node("Content 1", 0.9),
            self._create_node("Content 2", 0.8),
        ]
        result = RetrievalResult(nodes=nodes, query_text="test")

        assert result.has_results is True
        assert result.top_node is not None
        assert result.top_node.score == 0.9

    def test_get_citations(self):
        """Test citation extraction from nodes."""
        nodes = [
            self._create_node(
                paper_id="paper-1",
                paper_title="Study on pH",
                score=0.9,
            ),
            self._create_node(
                paper_id="paper-1",  # Duplicate paper
                paper_title="Study on pH",
                score=0.85,
            ),
            self._create_node(
                paper_id="paper-2",
                paper_title="Vaginal Health",
                score=0.8,
            ),
        ]
        result = RetrievalResult(nodes=nodes, query_text="test")

        citations = result.get_citations()

        # Should deduplicate by paper_id
        assert len(citations) == 2
        assert citations[0]["paper_id"] == "paper-1"
        assert citations[1]["paper_id"] == "paper-2"


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_search_mode_values(self):
        """Test all search mode values exist."""
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.BM25.value == "bm25"
        assert SearchMode.HYBRID.value == "hybrid"


class TestRerankStrategy:
    """Tests for RerankStrategy enum."""

    def test_strategy_values(self):
        """Test all rerank strategy values exist."""
        assert RerankStrategy.NONE.value == "none"
        assert RerankStrategy.SCORE_DECAY.value == "score_decay"
        assert RerankStrategy.CHUNK_TYPE_BOOST.value == "chunk_type_boost"
        assert RerankStrategy.RECENCY_BOOST.value == "recency_boost"
        assert RerankStrategy.RRF.value == "rrf"


class TestMedicalPaperRetriever:
    """Tests for MedicalPaperRetriever class."""

    def test_init_with_defaults(self):
        """Test retriever initialization with defaults."""
        retriever = MedicalPaperRetriever()

        assert retriever.config is not None
        assert retriever.config.top_k == 10

    def test_init_with_custom_config(self):
        """Test retriever initialization with custom config."""
        config = RetrievalConfig(top_k=20, similarity_threshold=0.8)
        retriever = MedicalPaperRetriever(config=config)

        assert retriever.config.top_k == 20
        assert retriever.config.similarity_threshold == 0.8

    def test_sync_retrieve_raises_error(self):
        """Test that synchronous _retrieve raises NotImplementedError."""
        from llama_index.core.schema import QueryBundle

        retriever = MedicalPaperRetriever()
        query = QueryBundle(query_str="test query")

        with pytest.raises(NotImplementedError):
            retriever._retrieve(query)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_retriever_defaults(self):
        """Test create_retriever with defaults uses hybrid search."""
        retriever = create_retriever()

        assert isinstance(retriever, MedicalPaperRetriever)
        assert retriever.config.search_mode == SearchMode.HYBRID
        assert retriever.config.rerank_strategy == RerankStrategy.CHUNK_TYPE_BOOST

    def test_create_retriever_no_rerank(self):
        """Test create_retriever without reranking."""
        retriever = create_retriever(rerank=False)

        assert retriever.config.rerank_strategy == RerankStrategy.NONE

    def test_create_retriever_with_filters(self):
        """Test create_retriever with chunk type filters."""
        chunk_types = [ChunkType.ABSTRACT, ChunkType.RESULTS]
        retriever = create_retriever(top_k=5, chunk_types=chunk_types)

        assert retriever.config.top_k == 5
        assert retriever.config.chunk_types == chunk_types

    def test_create_retriever_with_custom_weights(self):
        """Test create_retriever with custom hybrid weights."""
        retriever = create_retriever(
            semantic_weight=0.6,
            bm25_weight=0.4,
        )

        assert retriever.config.semantic_weight == pytest.approx(0.6)
        assert retriever.config.bm25_weight == pytest.approx(0.4)

    def test_create_semantic_retriever(self):
        """Test create_semantic_retriever."""
        retriever = create_semantic_retriever(top_k=5)

        assert isinstance(retriever, MedicalPaperRetriever)
        assert retriever.config.search_mode == SearchMode.SEMANTIC
        assert retriever.config.top_k == 5

    def test_create_bm25_retriever(self):
        """Test create_bm25_retriever."""
        retriever = create_bm25_retriever(top_k=5)

        assert isinstance(retriever, MedicalPaperRetriever)
        assert retriever.config.search_mode == SearchMode.BM25
        assert retriever.config.top_k == 5

    def test_create_hybrid_retriever(self):
        """Test create_hybrid_retriever."""
        retriever = create_hybrid_retriever(
            top_k=10,
            semantic_weight=0.8,
            bm25_weight=0.2,
        )

        assert isinstance(retriever, MedicalPaperRetriever)
        assert retriever.config.search_mode == SearchMode.HYBRID
        assert retriever.config.semantic_weight == pytest.approx(0.8)
        assert retriever.config.bm25_weight == pytest.approx(0.2)

    def test_create_multi_query_retriever(self):
        """Test create_multi_query_retriever uses hybrid by default."""
        retriever = create_multi_query_retriever(top_k=10, num_queries=5)

        assert isinstance(retriever, MultiQueryRetriever)
        assert retriever.num_queries == 5
        assert retriever.base_retriever.config.search_mode == SearchMode.HYBRID

    def test_create_multi_query_retriever_semantic(self):
        """Test create_multi_query_retriever with semantic mode."""
        retriever = create_multi_query_retriever(
            top_k=10,
            num_queries=3,
            search_mode=SearchMode.SEMANTIC,
        )

        assert retriever.base_retriever.config.search_mode == SearchMode.SEMANTIC


class TestReranking:
    """Tests for reranking functionality."""

    def _create_search_result(
        self,
        chunk_type: str,
        score: float,
        chunk_id: uuid.UUID | None = None,
    ):
        """Helper to create a mock SearchResult."""
        from ingestion.storage.vector_store import SearchResult

        return SearchResult(
            chunk_id=chunk_id or uuid.uuid4(),
            paper_id=uuid.uuid4(),
            content="Test content",
            chunk_type=chunk_type,
            section_title=None,
            page_number=None,
            chunk_metadata={},
            score=score,
        )

    def test_score_decay_reranking(self):
        """Test score decay reranking."""
        config = RetrievalConfig(rerank_strategy=RerankStrategy.SCORE_DECAY)
        retriever = MedicalPaperRetriever(config=config)

        results = [
            self._create_search_result("abstract", 0.9),
            self._create_search_result("results", 0.85),
            self._create_search_result("methods", 0.8),
        ]

        reranked = retriever._rerank_score_decay(results, decay_factor=0.9)

        # First result should keep most of its score
        assert reranked[0].score == pytest.approx(0.9 * (0.9**0))
        # Second should be decayed
        assert reranked[1].score == pytest.approx(0.85 * (0.9**1))
        # Third should be more decayed
        assert reranked[2].score == pytest.approx(0.8 * (0.9**2))

    def test_chunk_type_boost_reranking(self):
        """Test chunk type boost reranking."""
        config = RetrievalConfig(rerank_strategy=RerankStrategy.CHUNK_TYPE_BOOST)
        retriever = MedicalPaperRetriever(config=config)

        # Start with methods having highest raw score
        results = [
            self._create_search_result("methods", 0.95),
            self._create_search_result("abstract", 0.90),
            self._create_search_result("results", 0.85),
        ]

        reranked = retriever._rerank_chunk_type_boost(results)

        # After boost, abstract should be higher due to 1.2x boost
        # Abstract: 0.90 * 1.2 = 1.08
        # Results: 0.85 * 1.15 = 0.9775
        # Methods: 0.95 * 0.9 = 0.855
        assert reranked[0].chunk_type == "abstract"
        assert reranked[1].chunk_type == "results"
        assert reranked[2].chunk_type == "methods"


class TestReciprocalRankFusion:
    """Tests for RRF (Reciprocal Rank Fusion) functionality."""

    def _create_search_result(
        self,
        chunk_id: uuid.UUID,
        chunk_type: str = "abstract",
        score: float = 0.9,
    ):
        """Helper to create a mock SearchResult."""
        from ingestion.storage.vector_store import SearchResult

        return SearchResult(
            chunk_id=chunk_id,
            paper_id=uuid.uuid4(),
            content="Test content",
            chunk_type=chunk_type,
            section_title=None,
            page_number=None,
            chunk_metadata={},
            score=score,
        )

    def test_rrf_combines_results(self):
        """Test that RRF combines results from both sources."""
        config = RetrievalConfig(search_mode=SearchMode.HYBRID)
        retriever = MedicalPaperRetriever(config=config)

        # Create unique chunk IDs
        id1, id2, id3, id4 = [uuid.uuid4() for _ in range(4)]

        semantic_results = [
            self._create_search_result(id1, score=0.9),
            self._create_search_result(id2, score=0.8),
        ]
        bm25_results = [
            self._create_search_result(id3, score=0.85),
            self._create_search_result(id4, score=0.75),
        ]

        fused = retriever._reciprocal_rank_fusion(
            semantic_results, bm25_results
        )

        # All 4 unique chunks should be in results
        assert len(fused) == 4

    def test_rrf_deduplicates_overlapping_results(self):
        """Test that RRF deduplicates when same chunk appears in both."""
        config = RetrievalConfig(search_mode=SearchMode.HYBRID)
        retriever = MedicalPaperRetriever(config=config)

        # Create overlapping chunk IDs
        shared_id = uuid.uuid4()
        unique_id1 = uuid.uuid4()
        unique_id2 = uuid.uuid4()

        semantic_results = [
            self._create_search_result(shared_id, score=0.9),
            self._create_search_result(unique_id1, score=0.8),
        ]
        bm25_results = [
            self._create_search_result(shared_id, score=0.85),  # Same as semantic
            self._create_search_result(unique_id2, score=0.75),
        ]

        fused = retriever._reciprocal_rank_fusion(
            semantic_results, bm25_results
        )

        # Should have 3 unique chunks
        assert len(fused) == 3

        # Shared chunk should have highest RRF score (appears in both)
        chunk_ids = [r.chunk_id for r in fused]
        assert shared_id in chunk_ids

    def test_rrf_respects_weights(self):
        """Test that RRF respects semantic/BM25 weights."""
        config = RetrievalConfig(
            search_mode=SearchMode.HYBRID,
            semantic_weight=0.9,
            bm25_weight=0.1,
        )
        retriever = MedicalPaperRetriever(config=config)

        semantic_id = uuid.uuid4()
        bm25_id = uuid.uuid4()

        # Both ranked #1 in their respective lists
        semantic_results = [self._create_search_result(semantic_id, score=0.9)]
        bm25_results = [self._create_search_result(bm25_id, score=0.9)]

        fused = retriever._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            semantic_weight=0.9,
            bm25_weight=0.1,
        )

        # Semantic result should rank higher due to higher weight
        assert len(fused) == 2
        assert fused[0].chunk_id == semantic_id

    def test_rrf_with_empty_semantic(self):
        """Test RRF with empty semantic results."""
        config = RetrievalConfig(search_mode=SearchMode.HYBRID)
        retriever = MedicalPaperRetriever(config=config)

        bm25_results = [
            self._create_search_result(uuid.uuid4(), score=0.9),
        ]

        fused = retriever._reciprocal_rank_fusion([], bm25_results)

        assert len(fused) == 1

    def test_rrf_with_empty_bm25(self):
        """Test RRF with empty BM25 results."""
        config = RetrievalConfig(search_mode=SearchMode.HYBRID)
        retriever = MedicalPaperRetriever(config=config)

        semantic_results = [
            self._create_search_result(uuid.uuid4(), score=0.9),
        ]

        fused = retriever._reciprocal_rank_fusion(semantic_results, [])

        assert len(fused) == 1

