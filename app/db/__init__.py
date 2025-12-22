"""
Database module - models, connections, and repositories.

This module provides:
- Base: SQLAlchemy declarative base
- Models: User, HealthProfile, Paper, PaperChunk, QueryLog
- Session management: async engine, session factory, dependencies
- Enums: RiskLevel, ChunkType
"""

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.models import (
    ChunkType,
    HealthProfile,
    Paper,
    PaperChunk,
    QueryLog,
    RiskLevel,
    User,
)
from app.db.session import (
    async_session_factory,
    close_db,
    engine,
    get_async_session,
    get_session_context,
    init_db,
)

__all__ = [
    # Base classes
    "Base",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    # Models
    "User",
    "HealthProfile",
    "Paper",
    "PaperChunk",
    "QueryLog",
    # Enums
    "RiskLevel",
    "ChunkType",
    # Session management
    "engine",
    "async_session_factory",
    "get_async_session",
    "get_session_context",
    "init_db",
    "close_db",
]
