"""
Query Analyzer Node

Parses and analyzes user input to extract structured information:
- Parse and validate pH value
- Extract and categorize symptoms
- Identify medical concepts for search
- Generate optimized search queries
- Determine primary health concerns
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agent.prompts.guardrails import GUARDRAIL_SYSTEM_PROMPT_ADDITION
from agent.prompts.system_prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    QUERY_ANALYZER_USER_TEMPLATE,
)
from agent.state import AgentState, PHCategory, QueryAnalysis
from app.core.config import settings
from app.services.azure_openai import get_openai_client
from app.services.langfuse_client import get_langfuse_client

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def categorize_ph(ph_value: float) -> PHCategory:
    """
    Categorize pH value into risk categories.

    Args:
        ph_value: The pH value to categorize

    Returns:
        PHCategory enum value
    """
    if ph_value <= settings.ph_normal_max:
        return PHCategory.NORMAL
    elif ph_value <= settings.ph_concerning_threshold:
        return PHCategory.SLIGHTLY_ELEVATED
    elif ph_value <= 5.5:
        return PHCategory.ELEVATED
    else:
        return PHCategory.HIGHLY_ELEVATED


def extract_symptoms_from_profile(health_profile: dict[str, Any]) -> list[str]:
    """
    Extract symptoms from health profile.

    Args:
        health_profile: User's health profile dict

    Returns:
        List of symptom strings
    """
    symptoms = health_profile.get("symptoms", [])
    if isinstance(symptoms, list):
        return symptoms
    elif isinstance(symptoms, dict):
        # Handle case where symptoms is a dict with categories
        all_symptoms = []
        for category, symptom_list in symptoms.items():
            if isinstance(symptom_list, list):
                all_symptoms.extend(symptom_list)
        return all_symptoms
    return []


def build_default_search_queries(
    ph_value: float,
    ph_category: PHCategory,
    symptoms: list[str],
) -> list[str]:
    """
    Build default search queries based on pH and symptoms.

    Args:
        ph_value: The pH value
        ph_category: Categorized pH level
        symptoms: List of symptoms

    Returns:
        List of search query strings
    """
    queries = []

    # Base pH-related query
    if ph_category == PHCategory.NORMAL:
        queries.append("normal vaginal pH maintenance and healthy microbiome")
    elif ph_category == PHCategory.SLIGHTLY_ELEVATED:
        queries.append("slightly elevated vaginal pH causes and implications")
    elif ph_category == PHCategory.ELEVATED:
        queries.append("elevated vaginal pH bacterial vaginosis risk factors")
    else:
        queries.append("high vaginal pH infection risk clinical significance")

    # Add symptom-specific queries
    symptom_lower = [s.lower() for s in symptoms]

    if any("discharge" in s for s in symptom_lower):
        queries.append("vaginal discharge pH correlation clinical findings")

    if any("odor" in s or "smell" in s for s in symptom_lower):
        queries.append("vaginal odor causes pH imbalance research")

    if any("itch" in s for s in symptom_lower):
        queries.append("vaginal itching pH levels yeast infection bacterial vaginosis")

    if any("burn" in s or "pain" in s for s in symptom_lower):
        queries.append("vaginal burning discomfort pH research treatment")

    # Ensure we have at least 3 queries
    if len(queries) < 3:
        queries.append("vaginal health pH balance research evidence")

    return queries[:5]  # Limit to 5 queries


async def analyze_query_with_llm(
    ph_value: float,
    health_profile: dict[str, Any],
    query_text: str | None,
    trace=None,
) -> dict[str, Any]:
    """
    Use LLM to analyze the query and extract structured information.

    Args:
        ph_value: The pH value from the test strip
        health_profile: User's health profile
        query_text: Optional additional query text
        trace: Langfuse trace for observability

    Returns:
        Dict containing query analysis results
    """
    client = get_openai_client()

    # Build user message
    symptoms = health_profile.get("symptoms", [])
    if isinstance(symptoms, list):
        symptoms_str = ", ".join(symptoms) if symptoms else "None reported"
    else:
        symptoms_str = str(symptoms)

    user_message = QUERY_ANALYZER_USER_TEMPLATE.format(
        ph_value=ph_value,
        age=health_profile.get("age", "Not provided"),
        ethnicity=health_profile.get("ethnicity", "Not provided"),
        symptoms=symptoms_str,
        medical_history=json.dumps(health_profile.get("medical_history", {})),
        additional_info=json.dumps(health_profile.get("additional_info", {})),
        query_text=query_text or "None",
    )

    messages = [
        {
            "role": "system",
            "content": QUERY_ANALYZER_SYSTEM_PROMPT + GUARDRAIL_SYSTEM_PROMPT_ADDITION,
        },
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1000,
        )

        # Log to Langfuse if available
        if trace:
            langfuse = get_langfuse_client()
            langfuse.log_llm_call(
                trace=trace,
                name="query_analyzer",
                model=settings.azure_openai_deployment_name,
                input_messages=messages,
                output=response,
            )

        # Parse JSON response
        # Try to extract JSON from the response
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        return json.loads(response_text)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Query analysis LLM call failed: {e}")
        raise


def query_analyzer_node(state: AgentState) -> AgentState:
    """
    Synchronous wrapper for the query analyzer node.

    For LangGraph compatibility, this is the main entry point.
    """
    import asyncio

    return asyncio.run(aquery_analyzer_node(state))


async def aquery_analyzer_node(state: AgentState) -> AgentState:
    """
    Query Analyzer Node - Analyze user input and extract structured information.

    This node:
    1. Validates and categorizes the pH value
    2. Extracts symptoms from the health profile
    3. Uses LLM to identify medical concepts
    4. Generates optimized search queries

    Args:
        state: Current agent state

    Returns:
        Updated agent state with query_analysis
    """
    logger.info(f"Query Analyzer: Processing pH {state['ph_value']}")

    # Get trace for Langfuse
    trace = None
    langfuse = get_langfuse_client()
    if langfuse.is_configured() and state.get("langfuse_trace_id"):
        trace = langfuse.client.trace(id=state["langfuse_trace_id"])

    try:
        ph_value = state["ph_value"]
        health_profile = state.get("health_profile", {})
        query_text = state.get("query_text")

        # Categorize pH
        ph_category = categorize_ph(ph_value)

        # Extract symptoms from profile
        extracted_symptoms = extract_symptoms_from_profile(health_profile)

        # Try to get LLM analysis
        llm_analysis = {}
        try:
            if settings.is_azure_openai_configured():
                llm_analysis = await analyze_query_with_llm(
                    ph_value=ph_value,
                    health_profile=health_profile,
                    query_text=query_text,
                    trace=trace,
                )
        except Exception as e:
            logger.warning(f"LLM analysis failed, using defaults: {e}")
            state["errors"].append(f"LLM analysis failed: {str(e)}")

        # Build query analysis from LLM response or defaults
        medical_concepts = llm_analysis.get("medical_concepts", [
            "vaginal pH",
            "vaginal microbiome",
            "lactobacillus",
        ])

        search_queries = llm_analysis.get("search_queries")
        if not search_queries:
            search_queries = build_default_search_queries(
                ph_value, ph_category, extracted_symptoms
            )

        primary_concerns = llm_analysis.get("primary_concerns", [])
        if not primary_concerns:
            if ph_category != PHCategory.NORMAL:
                primary_concerns.append("Elevated pH level")
            if extracted_symptoms:
                primary_concerns.append("Active symptoms reported")

        urgency_indicators = llm_analysis.get("urgency_indicators", [])
        if ph_category in (PHCategory.ELEVATED, PHCategory.HIGHLY_ELEVATED):
            urgency_indicators.append("Significantly elevated pH")

        # Create QueryAnalysis
        query_analysis = QueryAnalysis(
            ph_value=ph_value,
            ph_category=ph_category,
            extracted_symptoms=(
                llm_analysis.get("extracted_symptoms") or extracted_symptoms
            ),
            medical_concepts=medical_concepts,
            search_queries=search_queries,
            primary_concerns=primary_concerns,
            urgency_indicators=urgency_indicators,
        )

        # Update state
        state["query_analysis"] = query_analysis.to_dict()
        state["retrieval_query_variations"] = search_queries

        logger.info(
            f"Query Analyzer complete: pH category={ph_category.value}, "
            f"queries={len(search_queries)}"
        )

    except Exception as e:
        logger.error(f"Query Analyzer failed: {e}")
        state["errors"].append(f"Query analyzer error: {str(e)}")

        # Set minimal analysis to allow workflow to continue
        state["query_analysis"] = {
            "ph_value": state["ph_value"],
            "ph_category": categorize_ph(state["ph_value"]).value,
            "extracted_symptoms": [],
            "medical_concepts": ["vaginal pH"],
            "search_queries": ["vaginal pH health research"],
            "primary_concerns": [],
            "urgency_indicators": [],
        }
        state["retrieval_query_variations"] = ["vaginal pH health research"]

    return state


