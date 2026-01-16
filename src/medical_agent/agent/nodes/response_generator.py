"""
Response Generator Node

Generates the final user-facing response with:
- Clear, empathetic summary of findings
- Evidence-based insights with citations
- Risk-appropriate messaging
- All necessary medical disclaimers
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from medical_agent.agent.prompts.disclaimers import (
    format_complete_disclaimer,
    format_response_footer,
)
from medical_agent.agent.prompts.guardrails import (
    GUARDRAIL_SYSTEM_PROMPT_ADDITION,
    validate_response,
    sanitize_response,
)
from medical_agent.agent.prompts.response_templates import (
    RiskLevel as TemplateRiskLevel,
    get_response_template,
    get_risk_config,
    get_ph_assessment_phrase,
    WELLNESS_TIPS_NORMAL,
    MONITORING_STEPS,
    DOCTOR_DISCUSSION_POINTS,
    EMERGENCY_DISCUSSION_POINTS,
)
from medical_agent.agent.prompts.system_prompts import (
    RESPONSE_GENERATOR_SYSTEM_PROMPT,
    RESPONSE_GENERATOR_USER_TEMPLATE,
)
from medical_agent.agent.state import AgentState, FinalResponse, RiskLevel
from medical_agent.core.config import settings
from medical_agent.infrastructure.azure_openai import get_openai_client
from medical_agent.infrastructure.langfuse_client import get_langfuse_client

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def format_citations_for_display(citations: list[dict[str, Any]]) -> list[str]:
    """
    Format citations for display in the response.

    Args:
        citations: List of citation dicts

    Returns:
        List of formatted citation strings
    """
    formatted = []
    for i, cit in enumerate(citations[:10], 1):  # Max 10 citations
        authors = cit.get("authors", "Unknown")
        title = cit.get("title", "Unknown")
        year = cit.get("year", "")
        doi = cit.get("doi", "")

        # Shorten author list
        if authors and "," in authors:
            first_author = authors.split(",")[0].strip()
            authors = f"{first_author} et al."

        citation_str = f"[{i}] {authors}"
        if year:
            citation_str += f" ({year})"
        citation_str += f". \"{title}\""
        if doi:
            citation_str += f". DOI: {doi}"

        formatted.append(citation_str)

    return formatted


def format_evidence_for_prompt(reasoning_output: dict[str, Any]) -> str:
    """
    Format evidence summary for the response generator prompt.

    Args:
        reasoning_output: The reasoning output dict

    Returns:
        Formatted evidence string
    """
    evidence = reasoning_output.get("evidence_summary", [])
    if not evidence:
        return "No specific research evidence was found for this query."

    lines = []
    for i, e in enumerate(evidence[:5], 1):
        finding = e.get("finding", "")
        source = e.get("source", "Unknown")
        relevance = e.get("relevance", "medium")
        confidence = e.get("confidence", "moderate")

        lines.append(f"{i}. Finding: {finding}")
        lines.append(f"   Source: {source} | Relevance: {relevance} | Confidence: {confidence}")

    return "\n".join(lines)


def format_insights_for_prompt(reasoning_output: dict[str, Any]) -> str:
    """
    Format synthesized insights for the response generator prompt.

    Args:
        reasoning_output: The reasoning output dict

    Returns:
        Formatted insights string
    """
    insights = reasoning_output.get("synthesized_insights", [])
    if not insights:
        return "Research-based insights will be provided by a healthcare provider."

    return "\n".join(f"- {insight}" for insight in insights)


async def generate_response_with_llm(
    state: AgentState,
    trace=None,
) -> dict[str, Any]:
    """
    Use LLM to generate the final formatted response.

    Args:
        state: Current agent state
        trace: Langfuse trace for observability

    Returns:
        Dict containing the formatted response
    """
    client = get_openai_client()

    health_profile = state.get("health_profile", {})
    risk_assessment = state.get("risk_assessment", {})
    reasoning_output = state.get("reasoning_output", {})

    # Format citations
    citations = reasoning_output.get("citations", [])
    formatted_citations = format_citations_for_display(citations)

    user_message = RESPONSE_GENERATOR_USER_TEMPLATE.format(
        age=health_profile.get("age", "Not provided"),
        ethnicity=health_profile.get("ethnicity", "Not provided"),
        ph_value=state.get("ph_value", 4.5),
        risk_level=risk_assessment.get("risk_level", "MONITOR"),
        insights=format_insights_for_prompt(reasoning_output),
        evidence_summary=format_evidence_for_prompt(reasoning_output),
        citations="\n".join(formatted_citations) if formatted_citations else "No citations available",
    )

    messages = [
        {
            "role": "system",
            "content": RESPONSE_GENERATOR_SYSTEM_PROMPT + GUARDRAIL_SYSTEM_PROMPT_ADDITION,
        },
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            temperature=0.5,  # Some creativity for natural language
            max_tokens=2500,
        )

        # Log to Langfuse
        if trace:
            langfuse = get_langfuse_client()
            langfuse.log_llm_call(
                trace=trace,
                name="response_generator",
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
        logger.warning(f"Failed to parse response generator output: {e}")
        return {}
    except Exception as e:
        logger.error(f"Response generation LLM call failed: {e}")
        raise


def build_templated_response(
    state: AgentState,
) -> dict[str, Any]:
    """
    Build response using templates when LLM is unavailable.

    Args:
        state: Current agent state

    Returns:
        Response dict
    """
    risk_assessment = state.get("risk_assessment", {})
    reasoning_output = state.get("reasoning_output", {})
    ph_value = state.get("ph_value", 4.5)

    # Get risk level
    risk_str = risk_assessment.get("risk_level", "MONITOR")
    try:
        risk_level = TemplateRiskLevel(risk_str.lower())
    except ValueError:
        risk_level = TemplateRiskLevel.MONITOR

    # Get risk config
    risk_config = get_risk_config(risk_level)

    # Build summary
    summary = f"{risk_config.emoji} Your vaginal pH of {ph_value} is {get_ph_assessment_phrase(risk_assessment.get('ph_category', 'unknown'))}."

    # Build main content
    ph_assessment = risk_assessment.get("ph_assessment", "")
    symptom_assessment = risk_assessment.get("symptom_assessment", "")

    main_content_parts = []
    if ph_assessment:
        main_content_parts.append(ph_assessment)
    if symptom_assessment:
        main_content_parts.append(symptom_assessment)

    insights = reasoning_output.get("synthesized_insights", [])
    if insights:
        main_content_parts.append("\n**Key Insights:**")
        for insight in insights[:3]:
            main_content_parts.append(f"- {insight}")

    main_content = "\n\n".join(main_content_parts)

    # Build next steps based on risk level
    if risk_level == TemplateRiskLevel.NORMAL:
        next_steps = WELLNESS_TIPS_NORMAL[:3]
    elif risk_level == TemplateRiskLevel.MONITOR:
        next_steps = MONITORING_STEPS[:3]
    elif risk_level == TemplateRiskLevel.CONCERNING:
        next_steps = DOCTOR_DISCUSSION_POINTS[:3]
    else:  # URGENT
        next_steps = EMERGENCY_DISCUSSION_POINTS[:3]

    # Format citations
    citations = reasoning_output.get("citations", [])
    formatted_citations = format_citations_for_display(citations)

    # Get disclaimer
    is_pregnant = state.get("is_pregnant", False)
    is_first_query = state.get("is_first_query", False)
    disclaimer = format_complete_disclaimer(
        risk_level=risk_str.lower(),
        is_pregnant=is_pregnant,
        is_first_query=is_first_query,
    )

    # Build risk level message
    risk_level_message = f"{risk_config.urgency_phrase}{risk_config.action_emphasis}."

    # Build full response
    full_response_parts = [
        f"## {risk_config.emoji} {risk_config.title}",
        "",
        f"**Your pH Level:** {ph_value}",
        "",
        main_content,
        "",
        "### Recommended Next Steps",
        "",
    ]
    for step in next_steps:
        full_response_parts.append(f"- {step}")

    if formatted_citations:
        full_response_parts.extend([
            "",
            "### References",
            "",
        ])
        for cit in formatted_citations[:5]:
            full_response_parts.append(cit)

    full_response_parts.extend([
        "",
        disclaimer,
    ])

    full_response = "\n".join(full_response_parts)

    return {
        "summary": summary,
        "main_content": main_content,
        "personalized_insights": insights[:3],
        "next_steps": next_steps,
        "risk_level_message": risk_level_message,
        "citations_formatted": formatted_citations,
        "disclaimers": disclaimer,
        "full_response": full_response,
    }


def response_generator_node(state: AgentState) -> AgentState:
    """
    Synchronous wrapper for the response generator node.

    For LangGraph compatibility, this is the main entry point.
    """
    import asyncio

    return asyncio.run(aresponse_generator_node(state))


async def aresponse_generator_node(state: AgentState) -> AgentState:
    """
    Response Generator Node - Create the final user-facing response.

    This node:
    1. Synthesizes all previous analysis into a coherent response
    2. Applies risk-appropriate tone and messaging
    3. Formats citations properly
    4. Adds required medical disclaimers
    5. Validates response against guardrails

    Args:
        state: Current agent state

    Returns:
        Updated agent state with final_response
    """
    logger.info("Response Generator: Creating final response")

    # Get trace for Langfuse
    trace = None
    langfuse = get_langfuse_client()
    if langfuse.is_configured() and state.get("langfuse_trace_id"):
        trace = langfuse.client.trace(id=state["langfuse_trace_id"])

    try:
        # Try LLM response generation
        llm_response = {}
        try:
            if settings.is_azure_openai_configured():
                llm_response = await generate_response_with_llm(state, trace)
        except Exception as e:
            logger.warning(f"LLM response generation failed, using templates: {e}")
            state["errors"].append(f"LLM response generation failed: {str(e)}")

        # Use LLM response or fall back to templates
        if llm_response and llm_response.get("full_response"):
            response_dict = llm_response
        else:
            response_dict = build_templated_response(state)

        # Ensure disclaimer is present
        if "disclaimer" not in response_dict.get("full_response", "").lower():
            risk_str = state.get("risk_assessment", {}).get("risk_level", "MONITOR")
            disclaimer = format_complete_disclaimer(
                risk_level=risk_str.lower(),
                is_pregnant=state.get("is_pregnant", False),
            )
            response_dict["full_response"] = (
                response_dict.get("full_response", "") + "\n\n" + disclaimer
            )
            response_dict["disclaimers"] = disclaimer

        # Validate response against guardrails
        full_response = response_dict.get("full_response", "")
        citations = state.get("reasoning_output", {}).get("citations", [])
        validation_result = validate_response(full_response, citations)

        if not validation_result.is_safe:
            logger.warning(
                f"Response failed guardrail validation: {len(validation_result.violations)} violations"
            )
            # Try to sanitize
            sanitized = sanitize_response(full_response)
            response_dict["full_response"] = sanitized

            # Log violations
            for violation in validation_result.violations:
                state["errors"].append(
                    f"Guardrail violation: {violation.violation_type.value} - {violation.description}"
                )

        # Add response footer with metadata
        footer = format_response_footer(
            response_id=state.get("session_id", "unknown"),
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            risk_level=state.get("risk_assessment", {}).get("risk_level", "MONITOR"),
        )
        response_dict["full_response"] = response_dict.get("full_response", "") + "\n\n" + footer

        # Create FinalResponse
        final_response = FinalResponse(
            summary=response_dict.get("summary", ""),
            main_content=response_dict.get("main_content", ""),
            personalized_insights=response_dict.get("personalized_insights", []),
            next_steps=response_dict.get("next_steps", []),
            risk_level_message=response_dict.get("risk_level_message", ""),
            citations_formatted=response_dict.get("citations_formatted", []),
            disclaimers=response_dict.get("disclaimers", ""),
            full_response=response_dict.get("full_response", ""),
        )

        state["final_response"] = final_response.to_dict()

        # Calculate total processing time
        if state.get("processing_start_time"):
            start = datetime.fromisoformat(state["processing_start_time"])
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            state["processing_time_ms"] = int(elapsed)

        logger.info(f"Response Generator complete: {len(final_response.full_response)} chars")

    except Exception as e:
        logger.error(f"Response Generator failed: {e}")
        state["errors"].append(f"Response generation error: {str(e)}")

        # Create minimal safe response
        risk_str = state.get("risk_assessment", {}).get("risk_level", "MONITOR")
        disclaimer = format_complete_disclaimer(risk_level=risk_str.lower())

        error_response = f"""## ⚠️ Unable to Generate Full Response

We encountered an issue analyzing your results. However, your safety is our priority.

**Your pH Level:** {state.get("ph_value", "N/A")}

Based on the available information, we recommend consulting with a healthcare provider for a proper evaluation.

{disclaimer}"""

        state["final_response"] = FinalResponse(
            summary="We encountered an issue generating your full response.",
            main_content="Please consult a healthcare provider for guidance.",
            personalized_insights=[],
            next_steps=["Consult a healthcare provider"],
            risk_level_message="We recommend professional consultation.",
            citations_formatted=[],
            disclaimers=disclaimer,
            full_response=error_response,
        ).to_dict()

    return state


