"""
Query API Routes

Provides the main endpoint for pH analysis queries using the
medical reasoning agent.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from medical_agent.agent.graph import run_medical_agent
from medical_agent.api.schemas import (
    CitationResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Analyze pH reading",
    description="""
    Submit a pH reading for analysis using the medical RAG agent.

    The agent will:
    1. Analyze the pH value and any symptoms
    2. Search relevant medical research papers
    3. Assess the risk level
    4. Generate evidence-based insights with citations

    **Important**: This is purely informational and NOT medical advice.
    Always consult a healthcare provider for medical concerns.
    """,
)
async def analyze_ph(request: QueryRequest) -> QueryResponse:
    """
    Analyze a pH reading and return evidence-based insights.

    Args:
        request: The query request with pH value and optional symptoms

    Returns:
        Analysis response with risk level, insights, and citations
    """
    # Verify Azure OpenAI is configured
    if not settings.is_azure_openai_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "SERVICE_NOT_CONFIGURED",
                "message": "Azure OpenAI is not configured. LLM features are unavailable.",
            },
        )

    logger.info(f"Processing query: pH={request.ph_value}")

    try:
        # Build comprehensive health profile from all provided data
        health_profile: dict[str, Any] = {}

        # Add basic info
        if request.age is not None:
            health_profile["age"] = request.age

        # Collect all symptoms from the symptoms object
        all_symptoms = []
        if request.symptoms:
            all_symptoms.extend(request.symptoms.discharge or [])
            all_symptoms.extend(request.symptoms.vulva_vagina or [])
            all_symptoms.extend(request.symptoms.smell or [])
            all_symptoms.extend(request.symptoms.urine or [])

        # Add medical history and context
        medical_history = {}

        if request.diagnoses:
            medical_history["diagnoses"] = request.diagnoses

        if request.ethnic_backgrounds:
            medical_history["ethnic_backgrounds"] = request.ethnic_backgrounds

        if request.menstrual_cycle:
            medical_history["menstrual_cycle"] = request.menstrual_cycle

        if request.birth_control:
            medical_history["birth_control"] = request.birth_control.model_dump(
                exclude_none=True, exclude_defaults=True
            )

        if request.hormone_therapy:
            medical_history["hormone_therapy"] = request.hormone_therapy

        if request.hrt:
            medical_history["hrt"] = request.hrt

        if request.fertility_journey:
            medical_history["fertility_journey"] = request.fertility_journey.model_dump(
                exclude_none=True, exclude_defaults=True
            )

        if all_symptoms:
            health_profile["symptoms"] = all_symptoms

        if medical_history:
            health_profile["medical_history"] = medical_history

        # Determine pregnancy status from fertility journey
        is_pregnant = False
        if (
            request.fertility_journey
            and request.fertility_journey.current_status == "I am pregnant"
        ):
            is_pregnant = True

        logger.info(
            f"User health profile: age={health_profile.get('age')}, "
            f"symptom_count={len(all_symptoms)}, "
            f"pregnant={is_pregnant}"
        )

        # Run the medical agent
        # NOTE: request.notes is intentionally NOT passed to the agent
        result = await run_medical_agent(
            ph_value=request.ph_value,
            health_profile=health_profile if health_profile else None,
            is_pregnant=is_pregnant,
        )

        # Extract response components
        final_response = result.get("final_response", {})
        reasoning_output = result.get("reasoning_output", {})

        # Build citations
        citations = []
        raw_citations = result.get("citations", []) or reasoning_output.get("citations", [])
        for c in raw_citations:
            if isinstance(c, dict):
                citations.append(
                    CitationResponse(
                        paper_id=c.get("paper_id", ""),
                        title=c.get("title"),
                        authors=c.get("authors"),
                        doi=c.get("doi"),
                        relevant_section=c.get("relevant_section"),
                    )
                )

        return QueryResponse(
            session_id=result.get("session_id", ""),
            ph_value=result.get("ph_value", request.ph_value),
            risk_level=result.get("risk_level", "UNKNOWN"),
            summary=final_response.get("summary", "Analysis complete."),
            main_content=final_response.get("main_content", ""),
            personalized_insights=final_response.get("personalized_insights", []),
            next_steps=final_response.get("next_steps", []),
            disclaimers=final_response.get(
                "disclaimers",
                "This information is for educational purposes only and is not medical advice. "
                "Please consult a healthcare provider for any health concerns.",
            ),
            citations=citations,
            processing_time_ms=result.get("processing_time_ms", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PROCESSING_ERROR",
                "message": f"Failed to process query: {str(e)}",
            },
        )
