"""
Metadata Filtering Helper for Vector Search

Provides SQL filter construction for querying chunks by extracted medical metadata.
Uses PostgreSQL JSONB operators to filter by ethnicities, diagnoses, symptoms, etc.
"""

from typing import Any

from sqlalchemy import and_
from sqlalchemy.sql import ColumnElement

from medical_agent.infrastructure.database.models import PaperChunk


def build_metadata_filters(
    filter_ethnicities: list[str] | None = None,
    filter_diagnoses: list[str] | None = None,
    filter_symptoms: list[str] | None = None,
    filter_menstrual_status: list[str] | None = None,
    filter_birth_control: list[str] | None = None,
    filter_hormone_therapy: list[str] | None = None,
    filter_fertility_treatments: list[str] | None = None,
) -> list[ColumnElement[bool]]:
    """
    Build SQLAlchemy filter conditions for medical metadata.

    Uses PostgreSQL JSONB operators to query the extracted_metadata field.
    Each filter checks if ANY of the provided values are present in the chunk's metadata array.

    Example SQL generated:
        chunk_metadata->'extracted_metadata'->'ethnicities' ?| array['African / Black', 'Asian']

    Args:
        filter_ethnicities: Filter by ethnicity values
        filter_diagnoses: Filter by diagnosis values
        filter_symptoms: Filter by symptom values
        filter_menstrual_status: Filter by menstrual status values
        filter_birth_control: Filter by birth control types
        filter_hormone_therapy: Filter by hormone therapy types
        filter_fertility_treatments: Filter by fertility treatment types

    Returns:
        List of SQLAlchemy filter conditions (combine with AND)
    """
    filters = []

    # Helper function to create JSONB array overlap filter
    def jsonb_array_contains_any(field_name: str, values: list[str]) -> ColumnElement[bool]:
        """Check if JSONB array field contains ANY of the given values."""
        # PostgreSQL operator: ?| checks if any array element exists
        # chunk_metadata->'extracted_metadata'->'field' ?| array['val1', 'val2']
        return PaperChunk.chunk_metadata["extracted_metadata"][field_name].op("?|")(values)

    if filter_ethnicities:
        filters.append(jsonb_array_contains_any("ethnicities", filter_ethnicities))

    if filter_diagnoses:
        filters.append(jsonb_array_contains_any("diagnoses", filter_diagnoses))

    if filter_symptoms:
        filters.append(jsonb_array_contains_any("symptoms", filter_symptoms))

    if filter_menstrual_status:
        filters.append(jsonb_array_contains_any("menstrual_status", filter_menstrual_status))

    if filter_birth_control:
        filters.append(jsonb_array_contains_any("birth_control", filter_birth_control))

    if filter_hormone_therapy:
        filters.append(jsonb_array_contains_any("hormone_therapy", filter_hormone_therapy))

    if filter_fertility_treatments:
        filters.append(
            jsonb_array_contains_any("fertility_treatments", filter_fertility_treatments)
        )

    return filters


def format_metadata_filters(
    filter_ethnicities: list[str] | None = None,
    filter_diagnoses: list[str] | None = None,
    filter_symptoms: list[str] | None = None,
    filter_menstrual_status: list[str] | None = None,
    filter_birth_control: list[str] | None = None,
    filter_hormone_therapy: list[str] | None = None,
    filter_fertility_treatments: list[str] | None = None,
) -> str:
    """
    Format metadata filters as a human-readable string for logging.

    Args:
        Same as build_metadata_filters

    Returns:
        Human-readable description of active filters
    """
    parts = []

    if filter_ethnicities:
        parts.append(f"ethnicities={filter_ethnicities}")
    if filter_diagnoses:
        parts.append(f"diagnoses={filter_diagnoses}")
    if filter_symptoms:
        parts.append(f"symptoms={filter_symptoms}")
    if filter_menstrual_status:
        parts.append(f"menstrual_status={filter_menstrual_status}")
    if filter_birth_control:
        parts.append(f"birth_control={filter_birth_control}")
    if filter_hormone_therapy:
        parts.append(f"hormone_therapy={filter_hormone_therapy}")
    if filter_fertility_treatments:
        parts.append(f"fertility_treatments={filter_fertility_treatments}")

    return ", ".join(parts) if parts else "no metadata filters"
