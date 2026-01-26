"""
Database Models for FemTech Medical RAG Agent

Defines all SQLAlchemy ORM models for the application including:
- Users and Health Profiles
- Medical Papers and Paper Chunks (with vector embeddings)
- Query Logs for compliance and history tracking
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from medical_agent.core.config import settings
from medical_agent.infrastructure.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    pass


# =============================================================================
# Enums
# =============================================================================


class ChunkType(str, Enum):
    """Types of chunks extracted from medical papers."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    TABLE = "table"
    FIGURE = "figure"
    REFERENCES = "references"
    OTHER = "other"


# =============================================================================
# User Models
# =============================================================================


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    User model - placeholder for authentication.
    
    Minimal user representation that will be extended when
    Zitadel authentication is integrated.
    """

    __tablename__ = "users"

    # Basic user information
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        comment="External auth provider ID (e.g., Zitadel)",
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )

    # Relationships
    health_profile: Mapped[Optional["HealthProfile"]] = relationship(
        "HealthProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    query_logs: Mapped[list["QueryLog"]] = relationship(
        "QueryLog",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class HealthProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    User health profile for personalized medical reasoning.
    
    Stores demographic and symptom information used to contextualize
    pH readings and provide relevant health insights.
    """

    __tablename__ = "health_profiles"

    # Foreign key to user
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # Demographics
    age: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    ethnicity: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # Health information stored as JSONB for flexibility
    symptoms: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Current symptoms as structured data",
    )
    medical_history: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Relevant medical history",
    )
    additional_info: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional health context",
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="health_profile",
    )


# =============================================================================
# Medical Paper Models
# =============================================================================


class Paper(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Medical research paper metadata and storage information.
    
    Stores paper metadata and references to the raw PDF in GCP storage.
    The actual content is chunked and stored in PaperChunk.
    """

    __tablename__ = "papers"

    # Paper metadata
    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    authors: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Comma-separated author names",
    )
    journal: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    publication_year: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )
    doi: Mapped[Optional[str]] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
    )
    abstract: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Storage information
    gcp_path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        unique=True,
        comment="Path to PDF in GCP Cloud Storage",
    )
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
        comment="SHA-256 hash for deduplication",
    )

    # Processing status
    is_processed: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        index=True,
    )
    processing_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Additional metadata as JSONB
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional paper metadata",
    )

    # Note: No direct relationship to chunks since paper_id is in metadata_ JSONB
    # To query chunks for a paper: query(PaperChunk).filter(PaperChunk.metadata_['paper_id'].astext == str(paper.id))


class PaperChunk(Base, TimestampMixin):
    """
    Chunked content from medical papers with vector embeddings.

    Pure LlamaIndex PGVectorStore schema (no custom columns):
    - id: BIGSERIAL primary key (auto-incrementing, LlamaIndex managed)
    - node_id: VARCHAR node identifier (UUID as string)
    - text: TEXT node content
    - metadata_: JSONB (contains paper_id, chunk_type, section_title, page_number,
                        extracted medical metadata, etc.)
    - embedding: VECTOR(3072) embeddings
    - text_search_tsv: TSVECTOR (generated column for hybrid search)
    - created_at, updated_at: timestamps

    All custom data (paper_id, chunk_type, citations, medical metadata) is stored
    in metadata_ JSONB following LlamaIndex pattern.

    Table name: data_paper_chunks (LlamaIndex uses 'data_' prefix)
    """

    __tablename__ = "data_paper_chunks"

    # Primary key (LlamaIndex manages this as auto-incrementing)
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Node identifier (LlamaIndex uses this for node.id_)
    node_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True,
        index=True,
    )

    # Chunk content (LlamaIndex standard)
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # All metadata in JSONB (LlamaIndex standard)
    # Contains: paper_id, chunk_type, chunk_index, section_title, page_number,
    # extracted_metadata (medical terms), table_summary, etc.
    metadata_: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="All chunk metadata including medical terms, paper references, and structural data",
    )

    # Vector embedding for similarity search
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True,
    )


# =============================================================================
# Query Log Model
# =============================================================================


class QueryLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Log of user queries for compliance, history, and improvement.
    
    Stores the complete context of each query including:
    - Input (pH value, health profile snapshot)
    - Processing (retrieved chunks, risk assessment)
    - Output (response, citations)
    - Feedback (optional user feedback)
    """

    __tablename__ = "query_logs"

    # Foreign key to user (nullable for anonymous queries)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Query input
    ph_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    query_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Additional user query text if provided",
    )
    health_profile_snapshot: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Snapshot of health profile at query time",
    )

    # Processing information
    retrieved_chunk_ids: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        comment="IDs of chunks retrieved for this query",
    )

    # Response
    response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    citations: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Paper citations used in response",
    )

    # Performance metrics
    processing_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # User feedback (optional)
    feedback_rating: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="User rating 1-5",
    )
    feedback_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="query_logs",
    )

    __table_args__ = (
        Index(
            "ix_query_logs_user_created",
            "user_id",
            "created_at",
        ),
    )

