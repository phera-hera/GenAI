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
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

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


# ============================================================================
# Query Generation and Rewriting Helpers
# ============================================================================


async def generate_initial_query(
    ph_value: float,
    health_profile: dict[str, Any],
    symptoms: list[str]
) -> str:
    """
    Generate initial search query from form data using LLM.

    Converts structured health data into a natural language medical research query
    optimized for retrieval.

    Args:
        ph_value: Vaginal pH reading
        health_profile: User's health context (diagnoses, age, etc.)
        symptoms: List of reported symptoms

    Returns:
        Generated search query string
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    # Build context string
    diagnoses_str = ", ".join(health_profile.get("diagnoses", []))
    symptoms_str = ", ".join(symptoms[:3]) if symptoms else "none reported"
    age_str = f"{health_profile.get('age')} years old" if health_profile.get("age") else "adult"

    prompt = f"""You are a medical research query generator. Create a focused, natural language search query for medical literature retrieval.

Patient Context:
- Vaginal pH: {ph_value}
- Age: {age_str}
- Diagnoses: {diagnoses_str if diagnoses_str else "none"}
- Symptoms: {symptoms_str}

Generate a 1-2 sentence medical research query that would retrieve relevant scientific literature about this presentation. Focus on the clinical significance and implications.

Do NOT include phrases like "research about" or "studies on". Write it as a direct clinical question.

Query:"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    query = response.content.strip()

    logger.info(f"Generated initial query: {query[:100]}...")
    return query


async def rewrite_followup_query(
    user_message: str,
    chat_history: list,
    ph_value: float,
    health_profile: dict[str, Any]
) -> str:
    """
    Rewrite follow-up query with conversation history context.

    Transforms context-dependent follow-up questions into standalone search queries
    by incorporating relevant information from conversation history and user profile.

    Args:
        user_message: Current user question
        chat_history: Previous conversation messages
        ph_value: User's pH value
        health_profile: User's health context

    Returns:
        Rewritten standalone search query
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    # Format conversation history (last 2 turns = 4 messages)
    history_text = "\n".join([
        f"{msg.type if hasattr(msg, 'type') else 'message'}: {str(msg.content if hasattr(msg, 'content') else msg)[:150]}"
        for msg in chat_history[-4:]
    ])

    # Build profile context
    diagnoses_str = ", ".join(health_profile.get("diagnoses", []))

    prompt = f"""Given this conversation history and user profile, rewrite the current question as a clear, standalone medical research query.

Conversation History:
{history_text}

User Profile:
- pH: {ph_value}
- Diagnoses: {diagnoses_str if diagnoses_str else "none"}

Current Question: "{user_message}"

Task: Rewrite this as a standalone search query for medical literature. Include relevant context from the conversation and profile ONLY if it helps clarify the question. If the question is already clear and complete, return it as-is.

Rewritten Query:"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()

    logger.info(f"Rewritten query: '{user_message}' → '{rewritten[:100]}...'")
    return rewritten


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

        # Add notes from symptoms
        if request.symptoms and request.symptoms.notes:
            health_profile["notes"] = request.symptoms.notes

        # Add medical context
        if request.diagnoses:
            health_profile["diagnoses"] = request.diagnoses
        if request.ethnic_backgrounds:
            health_profile["ethnic_backgrounds"] = request.ethnic_backgrounds
        if request.menstrual_cycle:
            health_profile["menstrual_cycle"] = request.menstrual_cycle

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

        # Fertility journey
        if request.fertility_journey:
            fertility_info = []
            if request.fertility_journey.current_status:
                fertility_info.append(request.fertility_journey.current_status)
            fertility_info.extend(request.fertility_journey.fertility_treatments or [])
            if fertility_info:
                health_profile["fertility_journey"] = fertility_info

        logger.info(f"Health profile: age={health_profile.get('age')}, symptoms={len(symptoms)}")

        # Use provided session_id or generate new one
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        logger.info(f"Session ID: {session_id}")

        # Generate or rewrite query based on interaction type
        if request.user_message:
            # FOLLOW-UP: Load conversation history and rewrite query
            logger.info("Follow-up question detected - loading conversation history")

            try:
                # Load conversation state from LangGraph memory
                state_snapshot = medical_rag_app.get_state(
                    config={"configurable": {"thread_id": session_id}}
                )
                chat_history = state_snapshot.values.get("messages", [])

                logger.info(f"Loaded {len(chat_history)} messages from conversation history")

                # Rewrite query with conversation context
                query_text = await rewrite_followup_query(
                    user_message=request.user_message,
                    chat_history=chat_history,
                    ph_value=request.ph_value,
                    health_profile=health_profile
                )

            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}. Using raw query.")
                query_text = request.user_message

        else:
            # INITIAL: Generate query from form data using LLM
            logger.info("Initial interaction - generating query from form data")

            query_text = await generate_initial_query(
                ph_value=request.ph_value,
                health_profile=health_profile,
                symptoms=symptoms
            )

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

        # Extract citations and filter to only those actually used
        raw_citations = result.get("citations", [])
        used_citation_ids = set(result.get("used_citations", []))

        # Filter citations to only include those actually used in the response
        # If used_citations is empty, show no citations (LLM found no relevant info)
        citations = []
        for c in raw_citations:
            citation_id = c.get("id")
            if citation_id in used_citation_ids:
                citations.append(
                    CitationResponse(
                        paper_id=str(c.get("node_id", c.get("id", "unknown"))),
                        title=c.get("file") or "Unknown Paper",
                        authors=None,  # Not available in chunk metadata
                        doi=None,  # Not available in chunk metadata
                        relevant_section=c.get("preview", ""),
                    )
                )

        logger.info(f"Analysis complete: {len(citations)} citations used (of {len(raw_citations)} retrieved)")

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
