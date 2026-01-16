"""
Medical Reasoner Node

Analyzes retrieved research evidence and synthesizes personalized insights:
- Evidence analysis with source evaluation
- Profile correlation with user's demographics
- Conflict resolution between studies
- Insight synthesis grounded in research
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from medical_agent.agent.nodes.retriever import extract_citations_from_chunks, format_chunks_for_context
from medical_agent.agent.prompts.guardrails import GUARDRAIL_SYSTEM_PROMPT_ADDITION
from medical_agent.agent.prompts.system_prompts import REASONER_SYSTEM_PROMPT, REASONER_USER_TEMPLATE
from medical_agent.agent.state import (
    AgentState,
    Citation,
    ConflictingEvidence,
    EvidenceSummary,
    ReasoningOutput,
)
from medical_agent.core.config import settings
from medical_agent.infrastructure.azure_openai import get_openai_client
from medical_agent.infrastructure.langfuse_client import get_langfuse_client

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def format_research_chunks_for_reasoning(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks for the reasoning prompt.

    Includes full content with paper metadata for proper citation.

    Args:
        chunks: List of retrieved chunk dicts

    Returns:
        Formatted string for LLM context
    """
    if not chunks:
        return "No research papers were found relevant to this query."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        # Build paper reference
        title = chunk.get("paper_title", "Unknown Title")
        authors = chunk.get("paper_authors", "Unknown Authors")
        doi = chunk.get("paper_doi", "")

        lines.append(f"--- Research Excerpt {i} ---")
        lines.append(f"Paper: {title}")
        lines.append(f"Authors: {authors}")
        if doi:
            lines.append(f"DOI: {doi}")
        lines.append(f"Section: {chunk.get('section_title', chunk.get('chunk_type', 'Unknown'))}")
        lines.append(f"Relevance Score: {chunk.get('score', 0):.3f}")
        lines.append("")
        lines.append("Content:")
        lines.append(chunk.get("content", ""))
        lines.append("")

    return "\n".join(lines)


async def reason_with_llm(
    state: AgentState,
    trace=None,
) -> dict[str, Any]:
    """
    Use LLM to analyze evidence and synthesize insights.

    Args:
        state: Current agent state
        trace: Langfuse trace for observability

    Returns:
        Dict containing reasoning output
    """
    client = get_openai_client()

    health_profile = state.get("health_profile", {})
    query_analysis = state.get("query_analysis", {})
    risk_assessment = state.get("risk_assessment", {})
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Format symptoms
    symptoms = health_profile.get("symptoms", [])
    if isinstance(symptoms, list):
        symptoms_str = ", ".join(symptoms) if symptoms else "None reported"
    else:
        symptoms_str = str(symptoms)

    # Extract and format health profile data (same as in query_analyzer)
    age = health_profile.get("age", "Not provided")

    # Handle ethnic backgrounds (convert list to string)
    ethnic_backgrounds = health_profile.get("medical_history", {}).get("ethnic_backgrounds", [])
    if isinstance(ethnic_backgrounds, list):
        ethnicity = ", ".join(ethnic_backgrounds) if ethnic_backgrounds else "Not provided"
    else:
        ethnicity = str(ethnic_backgrounds) if ethnic_backgrounds else "Not provided"

    # Medical history includes diagnoses, hormonal info, fertility info, etc.
    medical_history = health_profile.get("medical_history", {})

    # Format research chunks
    research_chunks = format_research_chunks_for_reasoning(retrieved_chunks)

    user_message = REASONER_USER_TEMPLATE.format(
        age=age,
        ethnicity=ethnicity,
        symptoms=symptoms_str,
        medical_history=json.dumps(medical_history),
        ph_value=query_analysis.get("ph_value", state.get("ph_value")),
        ph_category=query_analysis.get("ph_category", "unknown"),
        risk_level=risk_assessment.get("risk_level", "MONITOR"),
        risk_factors=", ".join(risk_assessment.get("risk_factors", [])),
        research_chunks=research_chunks,
    )

    messages = [
        {
            "role": "system",
            "content": REASONER_SYSTEM_PROMPT + GUARDRAIL_SYSTEM_PROMPT_ADDITION,
        },
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            temperature=0.4,  # Slightly higher for more nuanced reasoning
            max_tokens=2000,
        )

        # Log to Langfuse
        if trace:
            langfuse = get_langfuse_client()
            langfuse.log_llm_call(
                trace=trace,
                name="medical_reasoner",
                model=settings.azure_openai_deployment_name,
                input_messages=messages,
                output=response,
            )

        # Parse JSON response
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        return json.loads(response_text)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse reasoning response: {e}")
        return {}
    except Exception as e:
        logger.error(f"Reasoning LLM call failed: {e}")
        raise


def build_default_reasoning(
    state: AgentState,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build default reasoning output when LLM is unavailable.

    Args:
        state: Current agent state
        chunks: Retrieved chunks

    Returns:
        Default reasoning output dict
    """
    query_analysis = state.get("query_analysis", {})
    risk_assessment = state.get("risk_assessment", {})
    ph_value = query_analysis.get("ph_value", state.get("ph_value", 4.5))

    # Build evidence summary from chunks
    evidence_summary = []
    for i, chunk in enumerate(chunks[:5], 1):
        evidence_summary.append({
            "finding": f"Research excerpt {i} from {chunk.get('paper_title', 'research paper')}",
            "source": chunk.get("paper_title", "Unknown"),
            "relevance": "high" if chunk.get("score", 0) > 0.7 else "medium",
            "confidence": "moderate",
        })

    # Build citations
    citations = extract_citations_from_chunks(chunks)

    # Default insights based on pH
    synthesized_insights = []
    if ph_value <= 4.5:
        synthesized_insights.append(
            "Your pH level is within the normal range, indicating a healthy vaginal environment."
        )
    else:
        synthesized_insights.append(
            "Your pH level is elevated above the normal range of 3.8-4.5."
        )

    synthesized_insights.append(
        "A healthcare provider can offer personalized guidance based on your complete health picture."
    )

    return {
        "evidence_summary": evidence_summary,
        "profile_correlations": [],
        "conflicting_evidence": [],
        "synthesized_insights": synthesized_insights,
        "knowledge_gaps": [
            "Individual health context should be evaluated by a healthcare provider."
        ],
        "citations": citations,
    }


def reasoner_node(state: AgentState) -> AgentState:
    """
    Synchronous wrapper for the reasoner node.

    For LangGraph compatibility, this is the main entry point.
    """
    import asyncio

    return asyncio.run(areasoner_node(state))


async def areasoner_node(state: AgentState) -> AgentState:
    """
    Medical Reasoner Node - Analyze evidence and synthesize insights.

    This node:
    1. Analyzes relevance of each retrieved chunk
    2. Correlates findings with user's profile
    3. Resolves conflicting evidence
    4. Synthesizes actionable insights
    5. Maintains strict grounding in research

    Args:
        state: Current agent state

    Returns:
        Updated agent state with reasoning_output
    """
    logger.info("Reasoner: Analyzing evidence and synthesizing insights")

    # Get trace for Langfuse
    trace = None
    langfuse = get_langfuse_client()
    if langfuse.is_configured() and state.get("langfuse_trace_id"):
        trace = langfuse.client.trace(id=state["langfuse_trace_id"])

    try:
        retrieved_chunks = state.get("retrieved_chunks", [])

        # Get LLM reasoning
        llm_reasoning = {}
        try:
            if settings.is_azure_openai_configured() and retrieved_chunks:
                llm_reasoning = await reason_with_llm(state, trace)
        except Exception as e:
            logger.warning(f"LLM reasoning failed, using defaults: {e}")
            state["errors"].append(f"LLM reasoning failed: {str(e)}")

        # Use LLM response or defaults
        if llm_reasoning:
            reasoning_dict = llm_reasoning
        else:
            reasoning_dict = build_default_reasoning(state, retrieved_chunks)

        # Parse evidence summaries
        evidence_summary = []
        for e in reasoning_dict.get("evidence_summary", []):
            if isinstance(e, dict):
                evidence_summary.append(
                    EvidenceSummary(
                        finding=e.get("finding", ""),
                        source=e.get("source", "Unknown"),
                        relevance=e.get("relevance", "medium"),
                        confidence=e.get("confidence", "moderate"),
                    )
                )

        # Parse conflicting evidence
        conflicting_evidence = []
        for c in reasoning_dict.get("conflicting_evidence", []):
            if isinstance(c, dict):
                conflicting_evidence.append(
                    ConflictingEvidence(
                        topic=c.get("topic", ""),
                        positions=c.get("positions", []),
                        resolution=c.get("resolution", ""),
                    )
                )

        # Parse citations
        citations = []
        for cit in reasoning_dict.get("citations", []):
            if isinstance(cit, dict):
                citations.append(
                    Citation(
                        paper_id=cit.get("paper_id", ""),
                        title=cit.get("title", "Unknown"),
                        authors=cit.get("authors", "Unknown"),
                        year=cit.get("year"),
                        doi=cit.get("doi"),
                        relevant_section=cit.get("relevant_section"),
                        score=cit.get("score"),
                    )
                )

        # If no citations from LLM, extract from chunks
        if not citations:
            chunk_citations = extract_citations_from_chunks(retrieved_chunks)
            for cit in chunk_citations:
                citations.append(
                    Citation(
                        paper_id=cit.get("paper_id", ""),
                        title=cit.get("title", "Unknown"),
                        authors=cit.get("authors", "Unknown"),
                        doi=cit.get("doi"),
                        score=cit.get("score"),
                    )
                )

        # Create ReasoningOutput
        reasoning_output = ReasoningOutput(
            evidence_summary=evidence_summary,
            profile_correlations=reasoning_dict.get("profile_correlations", []),
            conflicting_evidence=conflicting_evidence,
            synthesized_insights=reasoning_dict.get("synthesized_insights", []),
            knowledge_gaps=reasoning_dict.get("knowledge_gaps", []),
            citations=citations,
        )

        state["reasoning_output"] = reasoning_output.to_dict()

        logger.info(
            f"Reasoner complete: {len(evidence_summary)} evidence items, "
            f"{len(citations)} citations"
        )

    except Exception as e:
        logger.error(f"Reasoner failed: {e}")
        state["errors"].append(f"Reasoning error: {str(e)}")

        # Set minimal output to allow workflow to continue
        state["reasoning_output"] = ReasoningOutput(
            evidence_summary=[],
            profile_correlations=[],
            conflicting_evidence=[],
            synthesized_insights=[
                "Unable to fully analyze research evidence. Please consult a healthcare provider."
            ],
            knowledge_gaps=[],
            citations=[],
        ).to_dict()

    return state


