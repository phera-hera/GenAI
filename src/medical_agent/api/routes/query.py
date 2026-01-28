"""
Query API Routes

Provides the main endpoint for pH analysis queries using the
medical reasoning agent.
"""

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from medical_agent.agents import medical_rag_app
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

    start_time = time.time()

    try:
        # Build health profile from request
        health_profile: dict[str, Any] = {}

        if request.age is not None:
            health_profile["age"] = request.age

        # Collect symptoms
        symptoms = []
        if request.symptoms:
            symptoms.extend(request.symptoms.discharge or [])
            symptoms.extend(request.symptoms.vulva_vagina or [])
            symptoms.extend(request.symptoms.smell or [])
            symptoms.extend(request.symptoms.urine or [])
        if symptoms:
            health_profile["symptoms"] = symptoms

        # Add medical context
        if request.diagnoses:
            health_profile["diagnoses"] = request.diagnoses
        if request.ethnic_backgrounds:
            health_profile["ethnicity"] = request.ethnic_backgrounds
        if request.menstrual_cycle:
            health_profile["menstrual_status"] = request.menstrual_cycle

        # Birth control
        bc_list = []
        if request.birth_control:
            if request.birth_control.general:
                bc_list.append(request.birth_control.general)
            if request.birth_control.pill:
                bc_list.append(request.birth_control.pill)
            if request.birth_control.iud:
                bc_list.append(request.birth_control.iud)
            bc_list.extend(request.birth_control.other_methods or [])
            bc_list.extend(request.birth_control.permanent or [])
        if bc_list:
            health_profile["birth_control"] = bc_list

        # Hormone therapy
        hrt_list = []
        hrt_list.extend(request.hormone_therapy or [])
        hrt_list.extend(request.hrt or [])
        if hrt_list:
            health_profile["hormone_therapy"] = hrt_list

        logger.info(f"Health profile: age={health_profile.get('age')}, symptoms={len(symptoms)}")

        # Use provided session_id or generate new one
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        logger.info(f"Session ID: {session_id}")

        # Build query message
        if request.user_message:
            # Follow-up question - use user's actual message
            query_text = request.user_message
            logger.info(f"Follow-up query: {query_text[:100]}...")
        else:
            # Initial request - auto-generate query from pH value
            query_text = f"My vaginal pH is {request.ph_value}. What does this mean?"
            if symptoms:
                query_text += f" I'm experiencing: {', '.join(symptoms[:3])}."
            logger.info(f"Initial query: {query_text[:100]}...")

        # Prepare state for LangGraph
        initial_state = {
            "messages": [{"role": "user", "content": query_text}],
            "ph_value": request.ph_value,
            "health_profile": health_profile if health_profile else {},
        }

        # Run LangGraph workflow with session continuity
        logger.info(f"Invoking medical RAG graph for session: {session_id}")

        result = medical_rag_app.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

        # Extract response from result
        assistant_message = result["messages"][-1]
        agent_reply = assistant_message.get("content", "") if isinstance(assistant_message, dict) else getattr(assistant_message, "content", "")

        # Extract citations from result
        raw_citations = result.get("citations", [])

        # Format citations for response
        citations = []
        for c in raw_citations:
            # Map graph citations to CitationResponse schema
            citations.append(
                CitationResponse(
                    paper_id=str(c.get("node_id", c.get("id", "unknown"))),
                    title=c.get("file", "Unknown Paper"),
                    authors=None,  # Not available in chunk metadata
                    doi=None,  # Not available in chunk metadata
                    relevant_section=c.get("preview", ""),
                )
            )

        logger.info(f"Analysis complete: {len(citations)} citations")

        # Build medical disclaimer
        disclaimers = (
            "This analysis is for informational purposes only and does not constitute medical advice. "
            "Always consult with a qualified healthcare provider for medical concerns, diagnosis, or treatment."
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return QueryResponse(
            session_id=session_id,
            ph_value=request.ph_value,
            agent_reply=agent_reply,
            disclaimers=disclaimers,
            citations=citations,
            processing_time_ms=processing_time_ms,
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
