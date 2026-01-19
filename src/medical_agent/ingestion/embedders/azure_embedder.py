"""
Azure OpenAI Embedding Generator for Medical Paper Chunks

Provides embedding generation for chunked medical paper content using
Azure OpenAI's text-embedding-3-large model.

Features:
- Single and batch embedding generation (3072-dimensional vectors)
- Rate limiting and retry logic
- Progress tracking for large batches
- Token estimation and validation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from medical_agent.core.config import settings
from medical_agent.core.exceptions import LLMError
from medical_agent.infrastructure.azure_openai import AzureOpenAIClient, get_openai_client

if TYPE_CHECKING:
    from medical_agent.ingestion.chunkers import ChunkedSection

logger = logging.getLogger(__name__)


# Azure OpenAI embedding limits
MAX_TOKENS_PER_INPUT = 8191  # text-embedding-3-large limit (same as 3-small)
MAX_BATCH_SIZE = 100  # Azure OpenAI batch limit
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimation


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector attached."""

    chunk: ChunkedSection
    embedding: list[float]
    token_count: int = 0

    @property
    def content(self) -> str:
        """Shortcut to chunk content."""
        return self.chunk.content


@dataclass
class EmbeddingResult:
    """Result of an embedding batch operation."""

    embedded_chunks: list[EmbeddedChunk] = field(default_factory=list)
    failed_chunks: list[tuple[ChunkedSection, str]] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def success_count(self) -> int:
        return len(self.embedded_chunks)

    @property
    def failure_count(self) -> int:
        return len(self.failed_chunks)

    @property
    def all_successful(self) -> bool:
        return len(self.failed_chunks) == 0


class AzureEmbedder:
    """
    Generate embeddings for medical paper chunks using Azure OpenAI.

    Handles batching, rate limiting, and error recovery for embedding
    generation at scale.

    Args:
        client: Optional AzureOpenAIClient instance. Creates one if not provided.
        batch_size: Number of chunks to embed per API call (max 100).
        max_retries: Maximum retry attempts per batch.
    """

    def __init__(
        self,
        client: AzureOpenAIClient | None = None,
        batch_size: int = 50,
        max_retries: int = 3,
    ):
        self.client = client or get_openai_client()
        self.batch_size = min(batch_size, MAX_BATCH_SIZE)
        self.max_retries = max_retries
        self._embedding_dimension = settings.embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension (3072 for text-embedding-3-large)."""
        return self._embedding_dimension

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a rough character-based estimation. For precise counts,
        use tiktoken with the specific model.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // CHARS_PER_TOKEN_ESTIMATE

    def validate_chunk(self, chunk: ChunkedSection) -> tuple[bool, str]:
        """
        Validate a chunk is suitable for embedding.

        Args:
            chunk: Chunk to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not chunk.content or not chunk.content.strip():
            return False, "Empty content"

        estimated_tokens = self.estimate_tokens(chunk.content)
        if estimated_tokens > MAX_TOKENS_PER_INPUT:
            return False, f"Content too long: ~{estimated_tokens} tokens (max {MAX_TOKENS_PER_INPUT})"

        return True, ""

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            LLMError: If embedding fails
        """
        if not text or not text.strip():
            raise LLMError("Cannot embed empty text")

        return self.client.generate_embedding(text.strip())

    def embed_chunk(self, chunk: ChunkedSection) -> EmbeddedChunk:
        """
        Generate embedding for a single chunk.

        Args:
            chunk: Chunk to embed

        Returns:
            EmbeddedChunk with embedding attached

        Raises:
            LLMError: If embedding fails
        """
        is_valid, error = self.validate_chunk(chunk)
        if not is_valid:
            raise LLMError(f"Invalid chunk: {error}")

        embedding = self.embed_single(chunk.content)
        token_count = self.estimate_tokens(chunk.content)

        return EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            token_count=token_count,
        )

    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _embed_batch_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.client.generate_embeddings_batch(texts, batch_size=len(texts))

    def embed_chunks(
        self,
        chunks: list[ChunkedSection],
        skip_invalid: bool = True,
        progress_callback: callable | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple chunks.

        Processes chunks in batches for efficiency while handling
        individual failures gracefully.

        Args:
            chunks: List of chunks to embed
            skip_invalid: If True, skip invalid chunks. If False, fail fast.
            progress_callback: Optional callback(processed, total) for progress updates

        Returns:
            EmbeddingResult with successful and failed chunks

        Raises:
            LLMError: If skip_invalid is False and an invalid chunk is found
        """
        result = EmbeddingResult()

        # Validate and filter chunks
        valid_chunks: list[ChunkedSection] = []
        for chunk in chunks:
            is_valid, error = self.validate_chunk(chunk)
            if is_valid:
                valid_chunks.append(chunk)
            elif skip_invalid:
                result.failed_chunks.append((chunk, error))
                logger.warning(f"Skipping invalid chunk: {error}")
            else:
                raise LLMError(f"Invalid chunk: {error}")

        if not valid_chunks:
            logger.warning("No valid chunks to embed")
            return result

        # Process in batches
        total_batches = (len(valid_chunks) + self.batch_size - 1) // self.batch_size
        processed = 0

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(valid_chunks))
            batch_chunks = valid_chunks[start:end]

            try:
                # Extract texts
                texts = [c.content.strip() for c in batch_chunks]

                # Generate embeddings
                embeddings = self._embed_batch_texts(texts)

                # Create embedded chunks
                for chunk, embedding in zip(batch_chunks, embeddings, strict=True):
                    token_count = self.estimate_tokens(chunk.content)
                    embedded = EmbeddedChunk(
                        chunk=chunk,
                        embedding=embedding,
                        token_count=token_count,
                    )
                    result.embedded_chunks.append(embedded)
                    result.total_tokens += token_count

                processed += len(batch_chunks)
                if progress_callback:
                    progress_callback(processed, len(valid_chunks))

                logger.debug(
                    f"Embedded batch {batch_idx + 1}/{total_batches} "
                    f"({len(batch_chunks)} chunks)"
                )

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Add all chunks in failed batch to failures
                for chunk in batch_chunks:
                    result.failed_chunks.append((chunk, str(e)))

        logger.info(
            f"Embedding complete: {result.success_count} successful, "
            f"{result.failure_count} failed, {result.total_tokens} total tokens"
        )

        return result


class AsyncAzureEmbedder:
    """
    Async wrapper for AzureEmbedder for use in async contexts.

    Runs the synchronous embedding operations in a thread pool to avoid
    blocking the event loop.
    """

    def __init__(
        self,
        client: AzureOpenAIClient | None = None,
        batch_size: int = 50,
        max_retries: int = 3,
    ):
        self._sync_embedder = AzureEmbedder(
            client=client,
            batch_size=batch_size,
            max_retries=max_retries,
        )

    @property
    def embedding_dimension(self) -> int:
        return self._sync_embedder.embedding_dimension

    async def embed_single(self, text: str) -> list[float]:
        """Async wrapper for single text embedding."""
        return await asyncio.to_thread(self._sync_embedder.embed_single, text)

    async def embed_chunk(self, chunk: ChunkedSection) -> EmbeddedChunk:
        """Async wrapper for single chunk embedding."""
        return await asyncio.to_thread(self._sync_embedder.embed_chunk, chunk)

    async def embed_chunks(
        self,
        chunks: list[ChunkedSection],
        skip_invalid: bool = True,
        progress_callback: callable | None = None,
    ) -> EmbeddingResult:
        """Async wrapper for batch chunk embedding."""
        return await asyncio.to_thread(
            self._sync_embedder.embed_chunks,
            chunks,
            skip_invalid,
            progress_callback,
        )


# Module-level factory functions


_embedder: AzureEmbedder | None = None
_async_embedder: AsyncAzureEmbedder | None = None


def get_embedder() -> AzureEmbedder:
    """Get or create the global AzureEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = AzureEmbedder()
    return _embedder


def get_async_embedder() -> AsyncAzureEmbedder:
    """Get or create the global AsyncAzureEmbedder instance."""
    global _async_embedder
    if _async_embedder is None:
        _async_embedder = AsyncAzureEmbedder()
    return _async_embedder

