"""
Agent State Schema for LangGraph Workflow

Defines the state that flows through the medical reasoning agent:
- Input: pH value, health profile, optional query text
- Processing: Query analysis, retrieved chunks, risk assessment, reasoning
- Output: Formatted response with citations and disclaimers
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

class RiskLevel(str, Enum):
    """Risk assessment levels based on pH and symptoms."""

    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    CONCERNING = "CONCERNING"
    URGENT = "URGENT"


class PHCategory(str, Enum):
    """pH value categories."""

    NORMAL = "normal"
    SLIGHTLY_ELEVATED = "slightly_elevated"
    ELEVATED = "elevated"
    HIGHLY_ELEVATED = "highly_elevated"


@dataclass
class Citation:
    """Citation from a research paper."""

    paper_id: str
    title: str
    authors: str
    year: int | None = None
    doi: str | None = None
    relevant_section: str | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "relevant_section": self.relevant_section,
            "score": self.score,
        }


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store."""

    chunk_id: str
    paper_id: str
    content: str
    chunk_type: str
    score: float
    paper_title: str | None = None
    paper_authors: str | None = None
    paper_doi: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "score": self.score,
            "paper_title": self.paper_title,
            "paper_authors": self.paper_authors,
            "paper_doi": self.paper_doi,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceSummary:
    """A piece of evidence from research."""

    finding: str
    source: str
    relevance: str  # "high", "medium", "low"
    confidence: str  # "strong", "moderate", "limited"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding": self.finding,
            "source": self.source,
            "relevance": self.relevance,
            "confidence": self.confidence,
        }


@dataclass
class ConflictingEvidence:
    """Conflicting evidence from different sources."""

    topic: str
    positions: list[str]
    resolution: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "positions": self.positions,
            "resolution": self.resolution,
        }


# =============================================================================
# Health Profile Types
# =============================================================================


@dataclass
class HealthProfile:
    """User health profile for personalized reasoning."""

    age: int | None = None
    ethnicity: str | None = None
    symptoms: list[str] = field(default_factory=list)
    medical_history: dict[str, Any] = field(default_factory=dict)
    additional_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "ethnicity": self.ethnicity,
            "symptoms": self.symptoms,
            "medical_history": self.medical_history,
            "additional_info": self.additional_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthProfile":
        return cls(
            age=data.get("age"),
            ethnicity=data.get("ethnicity"),
            symptoms=data.get("symptoms", []),
            medical_history=data.get("medical_history", {}),
            additional_info=data.get("additional_info", {}),
        )


# =============================================================================
# Query Analysis Output
# =============================================================================


@dataclass
class QueryAnalysis:
    """Result of query analysis."""

    ph_value: float
    ph_category: PHCategory
    extracted_symptoms: list[str]
    medical_concepts: list[str]
    search_queries: list[str]
    primary_concerns: list[str]
    urgency_indicators: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ph_value": self.ph_value,
            "ph_category": self.ph_category.value,
            "extracted_symptoms": self.extracted_symptoms,
            "medical_concepts": self.medical_concepts,
            "search_queries": self.search_queries,
            "primary_concerns": self.primary_concerns,
            "urgency_indicators": self.urgency_indicators,
        }


# =============================================================================
# Risk Assessment Output
# =============================================================================


@dataclass
class RiskAssessment:
    """Result of risk assessment."""

    risk_level: RiskLevel
    risk_factors: list[str]
    ph_assessment: str
    symptom_assessment: str
    key_findings: list[str]
    recommended_action: str
    escalation_needed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "risk_factors": self.risk_factors,
            "ph_assessment": self.ph_assessment,
            "symptom_assessment": self.symptom_assessment,
            "key_findings": self.key_findings,
            "recommended_action": self.recommended_action,
            "escalation_needed": self.escalation_needed,
        }


# =============================================================================
# Reasoning Output
# =============================================================================


@dataclass
class ReasoningOutput:
    """Result of medical reasoning."""

    evidence_summary: list[EvidenceSummary]
    profile_correlations: list[str]
    conflicting_evidence: list[ConflictingEvidence]
    synthesized_insights: list[str]
    knowledge_gaps: list[str]
    citations: list[Citation]
    has_sufficient_evidence: bool = True  # Flag for insufficient information cases

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_summary": [e.to_dict() for e in self.evidence_summary],
            "profile_correlations": self.profile_correlations,
            "conflicting_evidence": [c.to_dict() for c in self.conflicting_evidence],
            "synthesized_insights": self.synthesized_insights,
            "knowledge_gaps": self.knowledge_gaps,
            "citations": [c.to_dict() for c in self.citations],
            "has_sufficient_evidence": self.has_sufficient_evidence,
        }


# =============================================================================
# Final Response
# =============================================================================


@dataclass
class FinalResponse:
    """Final formatted response."""

    summary: str
    main_content: str
    personalized_insights: list[str]
    next_steps: list[str]
    risk_level_message: str
    citations_formatted: list[str]
    disclaimers: str
    full_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "main_content": self.main_content,
            "personalized_insights": self.personalized_insights,
            "next_steps": self.next_steps,
            "risk_level_message": self.risk_level_message,
            "citations_formatted": self.citations_formatted,
            "disclaimers": self.disclaimers,
            "full_response": self.full_response,
        }


# =============================================================================
# Agent State (TypedDict for LangGraph)
# =============================================================================


class AgentState(TypedDict, total=False):
    """
    State schema for the medical reasoning agent.

    This flows through all nodes in the LangGraph workflow.
    Uses TypedDict for LangGraph compatibility.
    """

    # --- Input ---
    session_id: str
    user_id: str | None
    ph_value: float
    health_profile: dict[str, Any]  # HealthProfile as dict
    query_text: str | None
    is_pregnant: bool
    is_first_query: bool

    # --- Query Analysis ---
    query_analysis: dict[str, Any]  # QueryAnalysis as dict

    # --- Retrieval ---
    retrieved_chunks: list[dict[str, Any]]  # List of RetrievedChunk as dicts
    retrieval_query_variations: list[str]
    retrieval_quality: str  # "sufficient", "insufficient", "low_relevance", "none"
    retrieval_chunk_count: int  # Number of chunks retrieved
    retrieval_max_score: float  # Highest similarity score

    # --- Risk Assessment ---
    risk_assessment: dict[str, Any]  # RiskAssessment as dict

    # --- Reasoning ---
    reasoning_output: dict[str, Any]  # ReasoningOutput as dict

    # --- Final Response ---
    final_response: dict[str, Any]  # FinalResponse as dict

    # --- Metadata ---
    processing_start_time: str
    processing_time_ms: int
    errors: list[str]
    langfuse_trace_id: str | None


def create_initial_state(
    ph_value: float,
    health_profile: HealthProfile | dict[str, Any] | None = None,
    query_text: str | None = None,
    user_id: str | None = None,
    is_pregnant: bool = False,
    is_first_query: bool = False,
) -> AgentState:
    """
    Create an initial agent state with the given inputs.

    Args:
        ph_value: The pH value from the test strip
        health_profile: User's health profile
        query_text: Optional additional query text
        user_id: Optional user identifier
        is_pregnant: Whether the user is pregnant
        is_first_query: Whether this is the user's first query

    Returns:
        Initial AgentState ready for workflow execution
    """
    if health_profile is None:
        profile_dict = HealthProfile().to_dict()
    elif isinstance(health_profile, HealthProfile):
        profile_dict = health_profile.to_dict()
    else:
        profile_dict = health_profile

    return AgentState(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        ph_value=ph_value,
        health_profile=profile_dict,
        query_text=query_text,
        is_pregnant=is_pregnant,
        is_first_query=is_first_query,
        query_analysis={},
        retrieved_chunks=[],
        retrieval_query_variations=[],
        retrieval_quality="unknown",  # Initialize as unknown
        retrieval_chunk_count=0,
        retrieval_max_score=0.0,
        risk_assessment={},
        reasoning_output={},
        final_response={},
        processing_start_time=datetime.utcnow().isoformat(),
        processing_time_ms=0,
        errors=[],
        langfuse_trace_id=None,
    )


