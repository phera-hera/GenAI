"""
Utility functions for RAG processing.

Handles formatting of retrieved nodes for citation-based responses.
"""

from pathlib import Path
from typing import Any

from llama_index.core.schema import NodeWithScore


def format_retrieved_nodes(nodes: list[NodeWithScore]) -> tuple[str, list[dict[str, Any]]]:
    """
    Convert NodeWithScore objects into formatted citation text and metadata.

    Args:
        nodes: List of retrieved nodes with relevance scores

    Returns:
        Tuple of:
            - docs_text: Formatted string with [1]: [Paper:page]: text citations
            - citations: List of citation metadata dicts

    Example output:
        docs_text: "[1]: [Diabetes2024.pdf:p23]: Insulin resistance is a key factor...\n\n[2]: [EndoReview.pdf:p45]: Metformin treatment..."
        citations: [{"id": 1, "file": "Diabetes2024.pdf", "page": "23", "score": 0.87, "preview": "Insulin resistance..."}]
    """
    if not nodes:
        return "No relevant medical research documents found.", []

    docs_text_parts = []
    citations = []

    for i, node_score in enumerate(nodes, 1):
        node = node_score.node

        # Extract metadata from LlamaIndex node metadata
        # title can be None explicitly, so use `or` not .get() default
        title = node.metadata.get("title") or Path(node.metadata.get("gcp_path", "")).stem or "Unknown Paper"
        doc_items = node.metadata.get("doc_items", [])
        page_no = doc_items[0]["prov"][0]["page_no"] if doc_items and doc_items[0].get("prov") else "N/A"

        # Get full text
        chunk_text = node.text.strip()

        # Create preview (first 100 chars for citation display)
        preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text

        # Format: [1]: [Title:page]: full text
        docs_text_parts.append(
            f"[{i}]: [{title}:p{page_no}]: {chunk_text}"
        )

        # Store citation metadata
        citations.append({
            "id": i,
            "file": title,
            "page": str(page_no),
            "score": round(node_score.score, 3) if node_score.score else 0.0,
            "preview": preview,
            "node_id": node.node_id,
        })

    return "\n\n".join(docs_text_parts), citations


def build_health_context(ph_value: float, health_profile: dict[str, Any]) -> str:
    """
    Build a concise health context string from user profile.

    Args:
        ph_value: Measured pH value
        health_profile: Dict with age, symptoms, diagnoses, etc.

    Returns:
        Formatted context string for LLM prompt
    """
    context_parts = [f"pH Value: {ph_value}"]

    if age := health_profile.get("age"):
        context_parts.append(f"Age: {age}")

    if ethnic_backgrounds := health_profile.get("ethnic_backgrounds"):
        context_parts.append(f"Ethnicity: {', '.join(ethnic_backgrounds)}")

    if diagnoses := health_profile.get("diagnoses"):
        context_parts.append(f"Diagnoses: {', '.join(diagnoses)}")

    if menstrual_cycle := health_profile.get("menstrual_cycle"):
        context_parts.append(f"Menstrual Cycle: {menstrual_cycle}")

    if birth_control := health_profile.get("birth_control"):
        context_parts.append(f"Birth Control: {', '.join(birth_control)}")

    if hormone_therapy := health_profile.get("hormone_therapy"):
        context_parts.append(f"Hormone Therapy: {', '.join(hormone_therapy)}")

    if fertility_journey := health_profile.get("fertility_journey"):
        context_parts.append(f"Fertility Status: {', '.join(fertility_journey)}")

    if symptoms := health_profile.get("symptoms"):
        # Check if symptoms is a dict or list
        if isinstance(symptoms, dict):
            symptom_list = []
            for key, values in symptoms.items():
                if values:
                    symptom_list.extend(values)
            if symptom_list:
                context_parts.append(f"Symptoms: {', '.join(symptom_list)}")
        elif isinstance(symptoms, list) and symptoms:
            context_parts.append(f"Symptoms: {', '.join(symptoms)}")

    if notes := health_profile.get("notes"):
        context_parts.append(f"Additional Notes: {notes}")

    return "\n".join(context_parts)
