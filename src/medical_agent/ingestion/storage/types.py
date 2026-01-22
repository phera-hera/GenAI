"""
Storage type definitions for vector operations and search results.

This module contains the core types used across the storage layer,
extracted from the original vector_store.py for better organization.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from medical_agent.infrastructure.database.models import ChunkType


# Vector distance metrics supported by pgvector
DistanceMetric = Literal["cosine", "l2", "inner_product"]


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    content: str
    chunk_type: str
    section_title: str | None
    page_number: int | None
    chunk_metadata: dict[str, Any]
    score: float  # Similarity score (higher is more similar for cosine)
    paper_title: str | None = None
    paper_authors: str | None = None
    paper_doi: str | None = None

    @property
    def citation(self) -> str:
        """Generate a basic citation string."""
        parts = []
        if self.paper_authors:
            parts.append(self.paper_authors.split(",")[0].strip() + " et al.")
        if self.paper_title:
            parts.append(f'"{self.paper_title}"')
        if self.paper_doi:
            parts.append(f"DOI: {self.paper_doi}")
        return " - ".join(parts) if parts else f"Paper ID: {self.paper_id}"


@dataclass
class SearchQuery:
    """Configuration for a vector similarity search."""

    embedding: list[float]
    top_k: int = 10
    distance_metric: DistanceMetric = "cosine"
    paper_ids: list[uuid.UUID] | None = None
    chunk_types: list[ChunkType] | None = None
    min_score: float | None = None
    include_paper_metadata: bool = True

    # Medical metadata filters
    # Filter chunks by extracted medical metadata
    filter_ethnicities: list[str] | None = None
    filter_diagnoses: list[str] | None = None
    filter_symptoms: list[str] | None = None
    filter_menstrual_status: list[str] | None = None
    filter_birth_control: list[str] | None = None
    filter_hormone_therapy: list[str] | None = None
    filter_fertility_treatments: list[str] | None = None


@dataclass
class StorageResult:
    """Result of a storage operation."""

    stored_count: int = 0
    failed_count: int = 0
    chunk_ids: list[uuid.UUID] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_successful(self) -> bool:
        return self.failed_count == 0
