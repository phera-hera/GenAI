"""
Risk Assessor Node

Evaluates the user's pH reading and symptoms to determine risk level:
- NORMAL: pH 3.8-4.5, no symptoms
- MONITOR: pH 3.8-4.5 with mild symptoms, or pH 4.5-5.0 no symptoms
- CONCERNING: pH 4.5-5.0 with symptoms, or pH 5.0-5.5
- URGENT: pH > 5.0 with symptoms, or pH > 5.5, or severe symptoms
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from medical_agent.agent.nodes.retriever import format_chunks_for_context
from medical_agent.agent.prompts.guardrails import GUARDRAIL_SYSTEM_PROMPT_ADDITION
from medical_agent.agent.prompts.system_prompts import (
    RISK_ASSESSOR_SYSTEM_PROMPT,
    RISK_ASSESSOR_USER_TEMPLATE,
)
from medical_agent.agent.state import AgentState, RiskAssessment, RiskLevel
from medical_agent.core.config import settings
from medical_agent.infrastructure.azure_openai import get_openai_client
from medical_agent.infrastructure.langfuse_client import get_langfuse_client

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Severe symptoms that should trigger urgent risk level
SEVERE_SYMPTOMS = {
    "fever",
    "severe pain",
    "severe cramping",
    "heavy bleeding",
    "unusual bleeding",
    "foul odor",
    "strong odor",
    "pelvic pain",
    "abdominal pain",
}


def check_severe_symptoms(symptoms: list[str]) -> bool:
    """
    Check if any symptoms are considered severe.

    Args:
        symptoms: List of symptom strings

    Returns:
        True if any severe symptoms are present
    """
    symptoms_lower = [s.lower() for s in symptoms]

    for symptom in symptoms_lower:
        for severe in SEVERE_SYMPTOMS:
            if severe in symptom:
                return True

    return False


def calculate_base_risk_level(
    ph_value: float,
    symptoms: list[str],
    is_pregnant: bool = False,
) -> RiskLevel:
    """
    Calculate the base risk level from pH and symptoms.

    This follows the risk matrix from the plan:
    - NORMAL: pH 3.8-4.5, no symptoms
    - MONITOR: pH 3.8-4.5 with mild symptoms, or pH 4.5-5.0 no symptoms
    - CONCERNING: pH 4.5-5.0 with symptoms, or pH 5.0-5.5
    - URGENT: pH > 5.0 with symptoms, or pH > 5.5, or severe symptoms

    Args:
        ph_value: The pH value
        symptoms: List of symptoms
        is_pregnant: Whether the user is pregnant (escalates risk)

    Returns:
        RiskLevel enum value
    """
    has_symptoms = len(symptoms) > 0
    has_severe_symptoms = check_severe_symptoms(symptoms)

    # Severe symptoms always escalate to URGENT
    if has_severe_symptoms:
        return RiskLevel.URGENT

    # pH-based risk assessment
    if ph_value > 5.5:
        # Highly elevated - always URGENT
        return RiskLevel.URGENT
    elif ph_value > settings.ph_concerning_threshold:  # > 5.0
        # Elevated pH
        if has_symptoms:
            return RiskLevel.URGENT
        else:
            return RiskLevel.CONCERNING
    elif ph_value > settings.ph_normal_max:  # 4.5 - 5.0
        # Slightly elevated
        if has_symptoms:
            return RiskLevel.CONCERNING
        else:
            return RiskLevel.MONITOR
    else:
        # Normal range (3.8 - 4.5)
        if has_symptoms:
            return RiskLevel.MONITOR
        else:
            return RiskLevel.NORMAL

    # Pregnancy escalation
    if is_pregnant and risk_level in (RiskLevel.NORMAL, RiskLevel.MONITOR):
        # Be more conservative during pregnancy
        if risk_level == RiskLevel.NORMAL and has_symptoms:
            return RiskLevel.MONITOR
        elif risk_level == RiskLevel.MONITOR:
            return RiskLevel.CONCERNING

    return risk_level


def get_ph_assessment_text(ph_value: float) -> str:
    """
    Get human-readable pH assessment text.

    Args:
        ph_value: The pH value

    Returns:
        Assessment string
    """
    if ph_value < settings.ph_normal_min:
        return f"Your pH of {ph_value} is below the normal range (3.8-4.5), indicating the vaginal environment may be too acidic."
    elif ph_value <= settings.ph_normal_max:
        return f"Your pH of {ph_value} is within the normal healthy range (3.8-4.5), indicating a balanced vaginal environment."
    elif ph_value <= settings.ph_concerning_threshold:
        return f"Your pH of {ph_value} is slightly elevated above the normal range (3.8-4.5). This may indicate changes in the vaginal microbiome."
    elif ph_value <= 5.5:
        return f"Your pH of {ph_value} is elevated above the normal range. This may be associated with bacterial changes that should be evaluated by a healthcare provider."
    else:
        return f"Your pH of {ph_value} is significantly elevated. This reading suggests you should consult with a healthcare provider for proper evaluation."


def get_symptom_assessment_text(symptoms: list[str]) -> str:
    """
    Get human-readable symptom assessment text.

    Args:
        symptoms: List of symptoms

    Returns:
        Assessment string
    """
    if not symptoms:
        return "No symptoms reported, which is a positive indicator."

    num_symptoms = len(symptoms)
    symptom_list = ", ".join(symptoms[:5])  # First 5

    if check_severe_symptoms(symptoms):
        return f"You reported {num_symptoms} symptom(s) including potentially concerning signs ({symptom_list}). These symptoms warrant prompt medical attention."
    elif num_symptoms == 1:
        return f"You reported one symptom: {symptom_list}. This should be monitored."
    else:
        return f"You reported {num_symptoms} symptoms: {symptom_list}. These should be considered together with your pH reading."


async def assess_risk_with_llm(
    state: AgentState,
    trace=None,
) -> dict[str, Any]:
    """
    Use LLM to provide detailed risk assessment.

    Args:
        state: Current agent state
        trace: Langfuse trace for observability

    Returns:
        Dict containing risk assessment details
    """
    client = get_openai_client()

    query_analysis = state.get("query_analysis", {})
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Format symptoms
    symptoms = query_analysis.get("extracted_symptoms", [])
    symptoms_text = "\n".join(f"- {s}" for s in symptoms) if symptoms else "None reported"

    # Format primary concerns
    concerns = query_analysis.get("primary_concerns", [])
    concerns_text = "\n".join(f"- {c}" for c in concerns) if concerns else "None identified"

    # Format urgency indicators
    urgency = query_analysis.get("urgency_indicators", [])
    urgency_text = "\n".join(f"- {u}" for u in urgency) if urgency else "None"

    # Format research context
    research_context = format_chunks_for_context(retrieved_chunks, max_chunks=5)

    user_message = RISK_ASSESSOR_USER_TEMPLATE.format(
        ph_value=query_analysis.get("ph_value", state.get("ph_value")),
        ph_category=query_analysis.get("ph_category", "unknown"),
        symptoms=symptoms_text,
        primary_concerns=concerns_text,
        urgency_indicators=urgency_text,
        research_context=research_context,
    )

    messages = [
        {
            "role": "system",
            "content": RISK_ASSESSOR_SYSTEM_PROMPT + GUARDRAIL_SYSTEM_PROMPT_ADDITION,
        },
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )

        # Log to Langfuse
        if trace:
            langfuse = get_langfuse_client()
            langfuse.log_llm_call(
                trace=trace,
                name="risk_assessor",
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
        logger.warning(f"Failed to parse risk assessment response: {e}")
        return {}
    except Exception as e:
        logger.error(f"Risk assessment LLM call failed: {e}")
        raise


def risk_assessor_node(state: AgentState) -> AgentState:
    """
    Synchronous wrapper for the risk assessor node.

    For LangGraph compatibility, this is the main entry point.
    """
    import asyncio

    return asyncio.run(arisk_assessor_node(state))


async def arisk_assessor_node(state: AgentState) -> AgentState:
    """
    Risk Assessor Node - Evaluate pH and symptoms for risk level.

    This node:
    1. Calculates base risk level from pH and symptoms
    2. Uses LLM to provide detailed assessment
    3. Considers retrieved research context
    4. Determines if escalation is needed

    Args:
        state: Current agent state

    Returns:
        Updated agent state with risk_assessment
    """
    logger.info("Risk Assessor: Evaluating risk level")

    # Get trace for Langfuse
    trace = None
    langfuse = get_langfuse_client()
    if langfuse.is_configured() and state.get("langfuse_trace_id"):
        trace = langfuse.client.trace(id=state["langfuse_trace_id"])

    try:
        query_analysis = state.get("query_analysis", {})
        ph_value = query_analysis.get("ph_value", state.get("ph_value", 4.5))
        symptoms = query_analysis.get("extracted_symptoms", [])
        is_pregnant = state.get("is_pregnant", False)

        # Calculate base risk level
        base_risk_level = calculate_base_risk_level(
            ph_value=ph_value,
            symptoms=symptoms,
            is_pregnant=is_pregnant,
        )

        # Get LLM-enhanced assessment
        llm_assessment = {}
        try:
            if settings.is_azure_openai_configured():
                llm_assessment = await assess_risk_with_llm(state, trace)
        except Exception as e:
            logger.warning(f"LLM risk assessment failed, using base calculation: {e}")
            state["errors"].append(f"LLM risk assessment failed: {str(e)}")

        # Use LLM risk level if provided, but never downgrade from base level
        llm_risk_str = llm_assessment.get("risk_level", "").upper()
        llm_risk_level = None
        if llm_risk_str in RiskLevel.__members__:
            llm_risk_level = RiskLevel[llm_risk_str]

        # Take the higher of base and LLM risk levels
        if llm_risk_level:
            risk_levels_ordered = [RiskLevel.NORMAL, RiskLevel.MONITOR, RiskLevel.CONCERNING, RiskLevel.URGENT]
            base_idx = risk_levels_ordered.index(base_risk_level)
            llm_idx = risk_levels_ordered.index(llm_risk_level)
            final_risk_level = risk_levels_ordered[max(base_idx, llm_idx)]
        else:
            final_risk_level = base_risk_level

        # Build risk factors
        risk_factors = llm_assessment.get("risk_factors", [])
        if not risk_factors:
            if ph_value > settings.ph_normal_max:
                risk_factors.append(f"Elevated pH ({ph_value})")
            if symptoms:
                risk_factors.append(f"{len(symptoms)} symptom(s) reported")
            if is_pregnant:
                risk_factors.append("Pregnancy")

        # Create RiskAssessment
        risk_assessment = RiskAssessment(
            risk_level=final_risk_level,
            risk_factors=risk_factors,
            ph_assessment=(
                llm_assessment.get("ph_assessment") or get_ph_assessment_text(ph_value)
            ),
            symptom_assessment=(
                llm_assessment.get("symptom_assessment")
                or get_symptom_assessment_text(symptoms)
            ),
            key_findings=llm_assessment.get("key_findings", []),
            recommended_action=llm_assessment.get(
                "recommended_action",
                get_recommended_action(final_risk_level),
            ),
            escalation_needed=(
                llm_assessment.get("escalation_needed", False)
                or final_risk_level in (RiskLevel.CONCERNING, RiskLevel.URGENT)
            ),
        )

        state["risk_assessment"] = risk_assessment.to_dict()

        logger.info(f"Risk Assessor complete: risk_level={final_risk_level.value}")

    except Exception as e:
        logger.error(f"Risk Assessor failed: {e}")
        state["errors"].append(f"Risk assessment error: {str(e)}")

        # Set safe default (MONITOR) to allow workflow to continue
        state["risk_assessment"] = RiskAssessment(
            risk_level=RiskLevel.MONITOR,
            risk_factors=["Assessment error - defaulting to MONITOR"],
            ph_assessment=get_ph_assessment_text(state.get("ph_value", 4.5)),
            symptom_assessment="Unable to fully assess symptoms.",
            key_findings=[],
            recommended_action="Consider consulting a healthcare provider.",
            escalation_needed=True,
        ).to_dict()

    return state


def get_recommended_action(risk_level: RiskLevel) -> str:
    """
    Get the recommended action based on risk level.

    Args:
        risk_level: The assessed risk level

    Returns:
        Recommended action string
    """
    actions = {
        RiskLevel.NORMAL: "Continue with regular health monitoring and maintain current practices.",
        RiskLevel.MONITOR: "Track your symptoms over the next few days. If they persist or worsen, consider consulting a healthcare provider.",
        RiskLevel.CONCERNING: "We recommend scheduling an appointment with a healthcare provider for proper evaluation.",
        RiskLevel.URGENT: "Please contact a healthcare provider today for evaluation. If you experience severe symptoms, consider urgent care.",
    }
    return actions.get(risk_level, actions[RiskLevel.MONITOR])


