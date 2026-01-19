"""
Application Configuration Management

Uses Pydantic Settings for type-safe configuration with environment variable support.
All settings can be overridden via environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    app_name: str = Field(default="FemTech Medical RAG Agent")
    app_version: str = Field(default="0.1.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)

    # -------------------------------------------------------------------------
    # Server Settings
    # -------------------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # -------------------------------------------------------------------------
    # Database Settings (PostgreSQL + pgvector)
    # -------------------------------------------------------------------------
    postgres_user: str = Field(default="femtech")
    postgres_password: str = Field(default="femtech_dev_password")
    postgres_db: str = Field(default="femtech_medical")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    
    # Direct database URL (overrides individual postgres settings if set)
    database_url: str | None = Field(default=None)
    
    @computed_field
    @property
    def database_connection_string(self) -> str:
        """Get the database connection string."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @computed_field
    @property
    def sync_database_connection_string(self) -> str:
        """Get synchronous database connection string for Alembic."""
        if self.database_url:
            return self.database_url.replace("postgresql+asyncpg", "postgresql")
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -------------------------------------------------------------------------
    # Azure OpenAI Settings - LLM (Chat Completions)
    # -------------------------------------------------------------------------
    azure_openai_api_key: str = Field(default="")
    azure_openai_endpoint: str = Field(default="")
    azure_openai_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_deployment_name: str = Field(default="gpt-4o")
    azure_openai_mini_deployment_name: str = Field(default="gpt-4o-mini")

    # -------------------------------------------------------------------------
    # Azure OpenAI Settings - Embeddings (Separate Endpoint/Key)
    # -------------------------------------------------------------------------
    # Use these if embeddings are deployed in a different Azure OpenAI instance
    azure_openai_embedding_api_key: str = Field(default="")
    azure_openai_embedding_endpoint: str = Field(default="")
    azure_openai_embedding_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_embedding_deployment_name: str = Field(
        default="text-embedding-3-large"
    )

    # -------------------------------------------------------------------------
    # LlamaParser Settings (LVM Mode with Azure OpenAI)
    # -------------------------------------------------------------------------
    # LlamaParser now uses LVM mode with your Azure OpenAI GPT-4o model
    # No separate API key needed - uses Azure OpenAI credentials
    llama_cloud_api_key: str = Field(default="")  # Deprecated, kept for backward compat

    def is_llama_parser_configured(self) -> bool:
        """
        Check if LlamaParser LVM mode is properly configured.

        LVM mode requires Azure OpenAI to be configured (same as medical reasoning).
        """
        return bool(
            self.azure_openai_api_key
            and self.azure_openai_endpoint
            and self.azure_openai_api_version
            and self.azure_openai_deployment_name
        )

    # -------------------------------------------------------------------------
    # GCP Settings
    # -------------------------------------------------------------------------
    gcp_project_id: str = Field(default="")
    gcp_bucket_name: str = Field(default="femtech-medical-papers")
    gcp_credentials_path: str | None = Field(default=None)
    gcp_cloud_sql_instance: str | None = Field(default=None)
    
    def is_gcp_configured(self) -> bool:
        """Check if GCP is properly configured."""
        return bool(self.gcp_project_id)
    
    def is_gcp_storage_configured(self) -> bool:
        """Check if GCP Storage is properly configured."""
        return bool(self.gcp_project_id and self.gcp_bucket_name)

    # -------------------------------------------------------------------------
    # Langfuse Observability Settings
    # -------------------------------------------------------------------------
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # -------------------------------------------------------------------------
    # Vector Search Settings
    # -------------------------------------------------------------------------
    # text-embedding-3-large produces 3072-dimensional vectors
    embedding_dimension: int = Field(default=3072)
    vector_similarity_top_k: int = Field(default=10)

    # -------------------------------------------------------------------------
    # Medical Reasoning Settings (pH thresholds)
    # -------------------------------------------------------------------------
    ph_normal_min: float = Field(default=3.8)
    ph_normal_max: float = Field(default=4.5)
    ph_concerning_threshold: float = Field(default=5.0)

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    def is_azure_openai_configured(self) -> bool:
        """Check if Azure OpenAI LLM is properly configured."""
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    def is_azure_openai_embedding_configured(self) -> bool:
        """
        Check if Azure OpenAI embeddings are properly configured.

        Returns True if either:
        1. Separate embedding credentials are provided, OR
        2. Main Azure OpenAI credentials exist (fallback)
        """
        # Use separate embedding credentials if provided
        if self.azure_openai_embedding_api_key and self.azure_openai_embedding_endpoint:
            return True

        # Fall back to main credentials if separate ones not provided
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    def is_langfuse_configured(self) -> bool:
        """Check if Langfuse observability is configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    Clear cache with get_settings.cache_clear() if needed.
    """
    return Settings()


# Global settings instance for convenience
settings = get_settings()


