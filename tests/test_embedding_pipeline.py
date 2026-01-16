"""
Tests for the embedding pipeline and vector storage.

These tests verify:
1. Azure embedder functionality
2. Vector store operations
3. Pipeline integration
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medical_agent.infrastructure.database.models import ChunkType

# Test data
SAMPLE_EMBEDDING = [0.1] * 1536  # 1536-dimensional vector
SAMPLE_CONTENT = "This is a sample medical research paper content about vaginal pH levels."


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_openai_client():
    """Create a mock Azure OpenAI client."""
    client = MagicMock()
    client.is_configured.return_value = True
    client.generate_embedding.return_value = SAMPLE_EMBEDDING
    client.generate_embeddings_batch.return_value = [SAMPLE_EMBEDDING, SAMPLE_EMBEDDING]
    return client


@pytest.fixture
def sample_chunk():
    """Create a sample ChunkedSection."""
    from ingestion.chunkers import ChunkedSection

    return ChunkedSection(
        content=SAMPLE_CONTENT,
        chunk_type=ChunkType.ABSTRACT,
        chunk_index=0,
        section_title="Abstract",
        page_number=1,
        chunk_metadata={"source": "test"},
    )


@pytest.fixture
def sample_chunks():
    """Create multiple sample chunks."""
    from ingestion.chunkers import ChunkedSection

    return [
        ChunkedSection(
            content=f"Test content {i} about medical research.",
            chunk_type=ChunkType.RESULTS if i % 2 == 0 else ChunkType.METHODS,
            chunk_index=i,
            section_title=f"Section {i}",
            page_number=i + 1,
        )
        for i in range(5)
    ]


# =============================================================================
# AzureEmbedder Tests
# =============================================================================


class TestAzureEmbedder:
    """Tests for AzureEmbedder class."""

    def test_estimate_tokens(self, mock_openai_client):
        """Test token estimation."""
        from ingestion.embedders import AzureEmbedder

        embedder = AzureEmbedder(client=mock_openai_client)

        # Test with known text
        text = "Hello world"  # 11 chars / 4 ≈ 2 tokens
        assert embedder.estimate_tokens(text) == 2

        # Test with longer text
        long_text = "a" * 1000  # 1000 chars / 4 = 250 tokens
        assert embedder.estimate_tokens(long_text) == 250

    def test_validate_chunk_valid(self, mock_openai_client, sample_chunk):
        """Test chunk validation with valid content."""
        from ingestion.embedders import AzureEmbedder

        embedder = AzureEmbedder(client=mock_openai_client)
        is_valid, error = embedder.validate_chunk(sample_chunk)

        assert is_valid is True
        assert error == ""

    def test_validate_chunk_empty(self, mock_openai_client):
        """Test chunk validation with empty content."""
        from ingestion.chunkers import ChunkedSection
        from ingestion.embedders import AzureEmbedder

        embedder = AzureEmbedder(client=mock_openai_client)

        empty_chunk = ChunkedSection(
            content="",
            chunk_type=ChunkType.OTHER,
            chunk_index=0,
        )
        is_valid, error = embedder.validate_chunk(empty_chunk)

        assert is_valid is False
        assert "Empty content" in error

    def test_validate_chunk_too_long(self, mock_openai_client):
        """Test chunk validation with content exceeding token limit."""
        from ingestion.chunkers import ChunkedSection
        from ingestion.embedders import AzureEmbedder

        embedder = AzureEmbedder(client=mock_openai_client)

        # Create chunk with ~10000 tokens (40000 chars)
        long_chunk = ChunkedSection(
            content="a" * 40000,
            chunk_type=ChunkType.OTHER,
            chunk_index=0,
        )
        is_valid, error = embedder.validate_chunk(long_chunk)

        assert is_valid is False
        assert "too long" in error

    def test_embed_single(self, mock_openai_client):
        """Test single text embedding."""
        from ingestion.embedders import AzureEmbedder

        embedder = AzureEmbedder(client=mock_openai_client)
        embedding = embedder.embed_single("Test text")

        assert embedding == SAMPLE_EMBEDDING
        assert len(embedding) == 1536
        mock_openai_client.generate_embedding.assert_called_once()

    def test_embed_chunk(self, mock_openai_client, sample_chunk):
        """Test single chunk embedding."""
        from ingestion.embedders import AzureEmbedder

        embedder = AzureEmbedder(client=mock_openai_client)
        result = embedder.embed_chunk(sample_chunk)

        assert result.embedding == SAMPLE_EMBEDDING
        assert result.chunk == sample_chunk
        assert result.token_count > 0

    def test_embed_chunks_batch(self, mock_openai_client, sample_chunks):
        """Test batch chunk embedding."""
        from ingestion.embedders import AzureEmbedder

        # Mock to return correct number of embeddings
        mock_openai_client.generate_embeddings_batch.return_value = [
            SAMPLE_EMBEDDING for _ in sample_chunks
        ]

        embedder = AzureEmbedder(client=mock_openai_client, batch_size=10)
        result = embedder.embed_chunks(sample_chunks)

        assert result.success_count == len(sample_chunks)
        assert result.failure_count == 0
        assert result.all_successful is True
        assert len(result.embedded_chunks) == len(sample_chunks)

    def test_embed_chunks_with_invalid(self, mock_openai_client, sample_chunks):
        """Test batch embedding with some invalid chunks."""
        from ingestion.chunkers import ChunkedSection
        from ingestion.embedders import AzureEmbedder

        # Add an empty chunk
        invalid_chunk = ChunkedSection(
            content="   ",  # Whitespace only
            chunk_type=ChunkType.OTHER,
            chunk_index=99,
        )
        chunks_with_invalid = sample_chunks + [invalid_chunk]

        mock_openai_client.generate_embeddings_batch.return_value = [
            SAMPLE_EMBEDDING for _ in sample_chunks
        ]

        embedder = AzureEmbedder(client=mock_openai_client, batch_size=10)
        result = embedder.embed_chunks(chunks_with_invalid, skip_invalid=True)

        assert result.success_count == len(sample_chunks)
        assert result.failure_count == 1
        assert result.all_successful is False


# =============================================================================
# AsyncAzureEmbedder Tests
# =============================================================================


class TestAsyncAzureEmbedder:
    """Tests for AsyncAzureEmbedder class."""

    @pytest.mark.asyncio
    async def test_async_embed_single(self, mock_openai_client):
        """Test async single text embedding."""
        from ingestion.embedders import AsyncAzureEmbedder

        embedder = AsyncAzureEmbedder(client=mock_openai_client)
        embedding = await embedder.embed_single("Test text")

        assert embedding == SAMPLE_EMBEDDING

    @pytest.mark.asyncio
    async def test_async_embed_chunk(self, mock_openai_client, sample_chunk):
        """Test async single chunk embedding."""
        from ingestion.embedders import AsyncAzureEmbedder

        embedder = AsyncAzureEmbedder(client=mock_openai_client)
        result = await embedder.embed_chunk(sample_chunk)

        assert result.embedding == SAMPLE_EMBEDDING
        assert result.chunk == sample_chunk


# =============================================================================
# VectorStore Tests
# =============================================================================


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.mark.asyncio
    async def test_search_query_creation(self):
        """Test SearchQuery dataclass."""
        from ingestion.storage import SearchQuery

        query = SearchQuery(
            embedding=SAMPLE_EMBEDDING,
            top_k=5,
            distance_metric="cosine",
        )

        assert len(query.embedding) == 1536
        assert query.top_k == 5
        assert query.distance_metric == "cosine"

    def test_search_result_citation(self):
        """Test SearchResult citation generation."""
        from ingestion.storage import SearchResult

        result = SearchResult(
            chunk_id=uuid.uuid4(),
            paper_id=uuid.uuid4(),
            content="Test content",
            chunk_type="abstract",
            section_title="Abstract",
            page_number=1,
            chunk_metadata={},
            score=0.95,
            paper_title="Vaginal Health Study",
            paper_authors="Smith J, Doe A",
            paper_doi="10.1234/test",
        )

        citation = result.citation
        assert "Smith J" in citation
        assert "Vaginal Health Study" in citation
        assert "10.1234/test" in citation

    def test_storage_result_properties(self):
        """Test StorageResult properties."""
        from ingestion.storage import StorageResult

        result = StorageResult(
            stored_count=10,
            failed_count=0,
            chunk_ids=[uuid.uuid4() for _ in range(10)],
        )

        assert result.all_successful is True

        result.failed_count = 2
        assert result.all_successful is False


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from ingestion.pipeline import PipelineConfig

        config = PipelineConfig()

        assert config.max_chunk_chars == 1200
        assert config.include_tables is True
        assert config.embedding_batch_size == 50
        assert config.skip_invalid_chunks is True
        assert config.delete_existing_chunks is True


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_success_property(self):
        """Test success property logic."""
        from ingestion.pipeline import PipelineResult

        result = PipelineResult(
            paper_id=uuid.uuid4(),
            paper_title="Test Paper",
            gcp_path="papers/test.pdf",
            stored=True,
            failed_count=0,
        )
        assert result.success is True

        result.failed_count = 1
        assert result.success is False

        result.failed_count = 0
        result.stored = False
        assert result.success is False

    def test_summary(self):
        """Test summary generation."""
        from ingestion.pipeline import PipelineResult

        result = PipelineResult(
            paper_id=uuid.uuid4(),
            paper_title="Test Paper",
            gcp_path="papers/test.pdf",
            stored=True,
            chunk_count=10,
            embedded_count=10,
            stored_count=10,
            failed_count=0,
            total_time_ms=1234,
        )

        summary = result.summary()
        assert "SUCCESS" in summary
        assert "Test Paper" in summary
        assert "1234ms" in summary


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    def test_compute_file_hash(self):
        """Test file hash computation."""
        from ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()

        content = b"Test PDF content"
        hash1 = pipeline.compute_file_hash(content)
        hash2 = pipeline.compute_file_hash(content)
        hash3 = pipeline.compute_file_hash(b"Different content")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars


# =============================================================================
# Integration Tests (require database)
# =============================================================================


class TestVectorStoreIntegration:
    """Integration tests for VectorStore with database."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires database connection")
    async def test_store_and_search(self):
        """Test storing chunks and searching."""
        from ingestion import (
            AsyncAzureEmbedder,
            ChunkedSection,
            EmbeddedChunk,
            SearchQuery,
            VectorStore,
        )
        from medical_agent.infrastructure.database.session import get_session_context

        # This would require a real database and Azure OpenAI connection
        pass

