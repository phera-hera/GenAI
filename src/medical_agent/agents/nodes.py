"""
LangGraph node functions for medical RAG workflow.

Each node is a pure function that takes state and returns state updates.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from medical_agent.agents.llamaindex_retrieval import retrieve_nodes
from medical_agent.agents.prompts import (
    HIGH_CONFIDENCE_PROMPT,
    LOW_CONFIDENCE_PROMPT,
    MEDIUM_CONFIDENCE_PROMPT,
)
from medical_agent.agents.reranker import rerank_nodes_with_metadata
from medical_agent.agents.state import MedicalAgentState
from medical_agent.agents.utils import build_health_context, format_retrieved_nodes
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


class MedicalResponse(BaseModel):
    """Structured output schema for medical RAG responses."""

    response: str = Field(
        description="Answer based solely on the provided documents. Every claim must be cited with [1], [2], etc. Do not include information not found in the documents."
    )
    used_citations: list[int] = Field(
        description="List of citation numbers actually used in the response (e.g., [1, 2] if you cited [1] and [2])"
    )


def retrieve_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Retrieval node: Fetches relevant medical research chunks.

    Process:
        1. Extract user's latest message and save as original_query
        2. Use refined query from refinement_history if available (retry), else use message query (first call)
        3. Over-retrieve with raw query (no health context dilution)
        4. Rerank using cross-encoder + metadata weighting
        5. Format nodes as citation text with [1][2] markers
        6. Return docs_text + citations + original_query for next node

    Args:
        state: Current agent state with messages and health context

    Returns:
        State update with docs_text, citations, and original_query
    """
    logger.info("Executing retrieve_node")

    # Get the latest user message
    if not state.get("messages"):
        logger.warning("No messages in state")
        return {
            "docs_text": "No query provided.",
            "citations": [],
            "original_query": "",
        }

    # Extract message content (this is the original user query)
    last_message = state["messages"][-1]
    message_query = last_message.content if hasattr(last_message, "content") else str(last_message.get("content", ""))

    # Determine which query to use: refined (retry) or original (first call)
    refinement_history = state.get("refinement_history", [])
    if refinement_history:
        # Retry: use the last refined query
        user_query = refinement_history[-1]
        logger.info(f"Using refined query (retry {state.get('retry_count', 0)}): {user_query[:100]}...")
    else:
        # First call: use the message query
        user_query = message_query
        logger.info(f"Retrieving nodes for query: {user_query[:100]}...")

    # Over-retrieve for reranking (15 candidates)
    nodes = retrieve_nodes(query=user_query, similarity_top_k=15)

    # Extract user health profile from state for metadata-weighted reranking
    health_profile = state.get("health_profile", {})

    # Rerank with cross-encoder (70%) + metadata overlap (30%)
    nodes = rerank_nodes_with_metadata(
        query=user_query,
        nodes=nodes,
        user_profile=health_profile,
        top_k=5,
    )

    # Format nodes into citation text
    docs_text, citations = format_retrieved_nodes(nodes)

    logger.info(f"Retrieved {len(citations)} citations after reranking")

    return {
        "docs_text": docs_text,
        "citations": citations,
        "original_query": message_query,  # Always save the user's message as original_query
    }


def generate_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Generation node: Produces medical response with confidence-adaptive prompting.

    Three-tier prompt strategy based on retrieval confidence:
    - HIGH (≥0.75): Confident, direct tone
    - MEDIUM (0.50-0.75): Exploratory, measured tone
    - LOW (<0.50): Humble, cautious with warnings

    Process:
        1. Get LLM with structured output
        2. Determine confidence tier from confidence_score
        3. Select appropriate prompt template
        4. Include conversation history if present
        5. Generate response with inline citations
        6. Return assistant message with metadata

    Args:
        state: Current agent state with docs_text, confidence_score, conversation history

    Returns:
        State update with new assistant message and used_citations
    """
    logger.info("Executing generate_node with confidence-adaptive prompting")

    # Get Azure OpenAI LLM (LangChain)
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,  # Deterministic for medical responses
    )

    # Use structured output to get response + used citations
    structured_llm = llm.with_structured_output(MedicalResponse)

    # Extract context
    docs_text = state.get("docs_text", "")
    ph_value = state.get("ph_value", 0.0)
    health_profile = state.get("health_profile", {})
    health_context = build_health_context(ph_value, health_profile)
    confidence_score = state.get("confidence_score", 0.5)

    # Get conversation history for context (exclude last user message as we'll reference it)
    messages = state.get("messages", [])
    conversation_history = ""
    if len(messages) > 1:
        # Format previous messages (exclude the current user query)
        history_parts = []
        for msg in messages[:-1]:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            history_parts.append(f"{role.capitalize()}: {content}")
        conversation_history = "\n".join(history_parts)

    # Get current user query
    last_message = messages[-1]
    current_query = last_message.get("content", "") if isinstance(last_message, dict) else getattr(last_message, "content", "")

    # Select prompt tier based on confidence score
    if confidence_score >= 0.75:
        base_prompt = HIGH_CONFIDENCE_PROMPT
        prompt_tier = "HIGH"
    elif confidence_score >= 0.50:
        base_prompt = MEDIUM_CONFIDENCE_PROMPT
        prompt_tier = "MEDIUM"
    else:
        base_prompt = LOW_CONFIDENCE_PROMPT
        prompt_tier = "LOW"

    logger.info(f"Using {prompt_tier} confidence prompt (score: {confidence_score:.3f})")

    # Format the selected prompt with context
    system_prompt = base_prompt.format(
        health_context=health_context,
        docs_text=docs_text,
        query=current_query,
    )

    # Add conversation history if exists
    if conversation_history:
        system_prompt += f"\n\nPREVIOUS CONVERSATION:\n{conversation_history}\n"

    logger.info("Generating structured response with Azure OpenAI (LangChain)")

    # Generate structured response
    result: MedicalResponse = structured_llm.invoke([HumanMessage(content=system_prompt)])

    logger.info(f"Response generated (tier: {prompt_tier}, citations: {len(result.used_citations)})")

    # Return as assistant message with confidence metadata
    assistant_msg = AIMessage(
        content=result.response,
        metadata={
            "confidence_score": confidence_score,
            "confidence_tier": prompt_tier,
        }
    )

    return {
        "messages": [assistant_msg],
        "used_citations": result.used_citations
    }
