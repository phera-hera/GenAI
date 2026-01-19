"""
Azure OpenAI Client Configuration

Provides configured OpenAI clients for:
- Chat completions (GPT-4o, GPT-4o-mini)
- Embeddings (text-embedding-3-large)

Supports separate endpoints for embeddings and LLM if deployed in different
Azure OpenAI instances.
"""

import logging
from typing import Any

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from medical_agent.core.config import settings
from medical_agent.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """
    Client for Azure OpenAI services.

    Supports separate endpoints for embeddings and LLM if they are deployed
    in different Azure OpenAI instances.

    Provides methods for:
    - Chat completions for medical reasoning
    - Embeddings for vector search
    """

    def __init__(
        self,
        # LLM settings
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        chat_deployment: str | None = None,
        # Embedding settings (can be different from LLM)
        embedding_api_key: str | None = None,
        embedding_endpoint: str | None = None,
        embedding_api_version: str | None = None,
        embedding_deployment: str | None = None,
    ):
        """
        Initialize Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key for LLM
            endpoint: Azure OpenAI endpoint URL for LLM
            api_version: API version for LLM
            chat_deployment: Deployment name for chat completions
            embedding_api_key: Azure OpenAI API key for embeddings (optional, uses api_key if not provided)
            embedding_endpoint: Azure OpenAI endpoint URL for embeddings (optional, uses endpoint if not provided)
            embedding_api_version: API version for embeddings (optional, uses api_version if not provided)
            embedding_deployment: Deployment name for embeddings
        """
        # LLM credentials
        self.api_key = api_key or settings.azure_openai_api_key
        self.endpoint = endpoint or settings.azure_openai_endpoint
        self.api_version = api_version or settings.azure_openai_api_version
        self.chat_deployment = chat_deployment or settings.azure_openai_deployment_name

        # Embedding credentials (with fallback to LLM credentials)
        self.embedding_api_key = (
            embedding_api_key
            or settings.azure_openai_embedding_api_key
            or self.api_key
        )
        self.embedding_endpoint = (
            embedding_endpoint
            or settings.azure_openai_embedding_endpoint
            or self.endpoint
        )
        self.embedding_api_version = (
            embedding_api_version
            or settings.azure_openai_embedding_api_version
            or self.api_version
        )
        self.embedding_deployment = (
            embedding_deployment or settings.azure_openai_embedding_deployment_name
        )

        self._llm_client: AzureOpenAI | None = None
        self._embedding_client: AzureOpenAI | None = None

    @property
    def llm_client(self) -> AzureOpenAI:
        """Get or create the LLM Azure OpenAI client."""
        if self._llm_client is None:
            if not self.is_llm_configured():
                raise LLMError("Azure OpenAI LLM is not configured")

            self._llm_client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )
        return self._llm_client

    @property
    def embedding_client(self) -> AzureOpenAI:
        """Get or create the embedding Azure OpenAI client."""
        if self._embedding_client is None:
            if not self.is_embedding_configured():
                raise LLMError("Azure OpenAI embedding is not configured")

            self._embedding_client = AzureOpenAI(
                api_key=self.embedding_api_key,
                api_version=self.embedding_api_version,
                azure_endpoint=self.embedding_endpoint,
            )
        return self._embedding_client

    @property
    def client(self) -> AzureOpenAI:
        """Get the LLM client (for backwards compatibility)."""
        return self.llm_client

    def is_llm_configured(self) -> bool:
        """Check if Azure OpenAI LLM is properly configured."""
        return bool(self.api_key and self.endpoint)

    def is_embedding_configured(self) -> bool:
        """Check if Azure OpenAI embeddings are properly configured."""
        return bool(self.embedding_api_key and self.embedding_endpoint)

    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured (both LLM and embeddings)."""
        return self.is_llm_configured() and self.is_embedding_configured()

    def verify_connection(self) -> bool:
        """
        Verify connection to Azure OpenAI LLM.

        Returns:
            True if connection is successful

        Raises:
            LLMError: If connection fails
        """
        if not self.is_llm_configured():
            raise LLMError("Azure OpenAI LLM is not configured")

        try:
            # Test with a minimal completion
            response = self.llm_client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            logger.info(
                f"Successfully connected to Azure OpenAI LLM "
                f"(endpoint: {self.endpoint}, deployment: {self.chat_deployment})"
            )
            return True
        except Exception as e:
            raise LLMError(f"Failed to connect to Azure OpenAI LLM: {e}")

    def verify_embedding_deployment(self) -> bool:
        """
        Verify the embedding deployment is working.

        Returns:
            True if successful

        Raises:
            LLMError: If verification fails
        """
        if not self.is_embedding_configured():
            raise LLMError("Azure OpenAI embeddings are not configured")

        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_deployment,
                input="test",
            )
            logger.info(
                f"Successfully connected to embedding model "
                f"(endpoint: {self.embedding_endpoint}, deployment: {self.embedding_deployment})"
            )
            return True
        except Exception as e:
            raise LLMError(f"Failed to connect to embedding deployment: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the API

        Returns:
            Generated text response

        Raises:
            LLMError: If generation fails
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise LLMError(f"Chat completion failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            LLMError: If embedding fails
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_deployment,
                input=text,
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise LLMError(f"Embedding generation failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate_embeddings_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors

        Raises:
            LLMError: If embedding fails
        """
        all_embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = self.embedding_client.embeddings.create(
                    model=self.embedding_deployment,
                    input=batch,
                )

                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i // batch_size + 1}")

            return all_embeddings

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise LLMError(f"Batch embedding generation failed: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a rough estimation (4 chars per token for English).
        For precise counts, use tiktoken with the specific model.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4


# Global client instance
_openai_client: AzureOpenAIClient | None = None


def get_openai_client() -> AzureOpenAIClient:
    """Get or create the global Azure OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AzureOpenAIClient()
    return _openai_client


# LlamaIndex integration helpers


def get_llama_index_llm():
    """
    Get a LlamaIndex-compatible Azure OpenAI LLM.

    Returns:
        AzureOpenAI LLM configured for LlamaIndex
    """
    from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI

    if not settings.is_azure_openai_configured():
        raise LLMError("Azure OpenAI LLM is not configured")

    return LlamaAzureOpenAI(
        model=settings.azure_openai_deployment_name,
        deployment_name=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def get_llama_index_embed_model():
    """
    Get a LlamaIndex-compatible Azure OpenAI embedding model.

    Uses separate embedding credentials if configured, otherwise falls back
    to main Azure OpenAI credentials.

    Returns:
        AzureOpenAIEmbedding configured for LlamaIndex
    """
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

    if not settings.is_azure_openai_embedding_configured():
        raise LLMError("Azure OpenAI embeddings are not configured")

    # Use separate embedding credentials if available
    api_key = (
        settings.azure_openai_embedding_api_key or settings.azure_openai_api_key
    )
    endpoint = (
        settings.azure_openai_embedding_endpoint or settings.azure_openai_endpoint
    )
    api_version = (
        settings.azure_openai_embedding_api_version
        or settings.azure_openai_api_version
    )

    return AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment_name,
        deployment_name=settings.azure_openai_embedding_deployment_name,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
