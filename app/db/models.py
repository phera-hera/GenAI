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

from app.core.config import settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    pass


# =============================================================================
# Enums
# =============================================================================


class RiskLevel(str, Enum):
    """Risk assessment levels based on pH and symptoms."""

    NORMAL = "normal"
    MONITOR = "monitor"
    CONCERNING = "concerning"
    URGENT = "urgent"


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

    # Relationships
    chunks: Mapped[list["PaperChunk"]] = relationship(
        "PaperChunk",
        back_populates="paper",
        cascade="all, delete-orphan",
    )


class PaperChunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Chunked content from medical papers with vector embeddings.
    
    Each paper is split into semantic chunks (abstract, sections, tables)
    and each chunk has an embedding for vector similarity search.
    """

    __tablename__ = "paper_chunks"

    # Foreign key to paper
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Chunk content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    chunk_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=ChunkType.OTHER.value,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Order of chunk within the paper",
    )

    # Vector embedding for similarity search
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True,
    )

    # Chunk metadata
    section_title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional chunk metadata (e.g., table structure)",
    )

    # Relationships
    paper: Mapped["Paper"] = relationship(
        "Paper",
        back_populates="chunks",
    )

    # Indexes for vector similarity search
    __table_args__ = (
        # IVFFlat index for approximate nearest neighbor search
        # Note: This is created via Alembic migration for better control
        Index(
            "ix_paper_chunks_paper_chunk_type",
            "paper_id",
            "chunk_type",
        ),
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
    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=RiskLevel.NORMAL.value,
        index=True,
    )
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
        Index(
            "ix_query_logs_risk_level_created",
            "risk_level",
            "created_at",
        ),
    )

