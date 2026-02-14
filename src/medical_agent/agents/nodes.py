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
from medical_agent.agents.reranker import rerank_nodes
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
        1. Extract user's latest message
        2. Over-retrieve with raw query (no health context dilution)
        3. Rerank using cross-encoder
        4. Format nodes as citation text with [1][2] markers
        5. Return docs_text + citations for next node

    Args:
        state: Current agent state with messages and health context

    Returns:
        State update with docs_text and citations
    """
    logger.info("Executing retrieve_node")

    # Get the latest user message
    if not state.get("messages"):
        logger.warning("No messages in state")
        return {
            "docs_text": "No query provided.",
            "citations": []
        }

    # Extract query from last message
    last_message = state["messages"][-1]
    user_query = last_message.content if hasattr(last_message, "content") else str(last_message.get("content", ""))

    # CHANGED: Use raw query for retrieval (no health context dilution)
    logger.info(f"Retrieving nodes for query: {user_query[:100]}...")

    # Over-retrieve for reranking (15 candidates)
    nodes = retrieve_nodes(query=user_query, similarity_top_k=15)

    # Rerank with cross-encoder and keep top-5
    nodes = rerank_nodes(query=user_query, nodes=nodes, top_k=5)

    # Format nodes into citation text
    docs_text, citations = format_retrieved_nodes(nodes)

    logger.info(f"Retrieved {len(citations)} citations after reranking")

    return {
        "docs_text": docs_text,
        "citations": citations
    }


def generate_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Generation node: Produces medical response with inline citations.

    Process:
        1. Get LLM from Azure OpenAI (LangChain) with structured output
        2. Build strict medical prompt with docs_text
        3. Include conversation history for context
        4. Generate response with inline [1][2] citations
        5. Return assistant message AND list of used citations

    Args:
        state: Current agent state with docs_text and conversation history

    Returns:
        State update with new assistant message and used_citations
    """
    logger.info("Executing generate_node")

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

    # Build generation prompt — optimized for faithfulness + answer relevancy
    system_prompt = f"""You are a caring and knowledgeable women's health consultant. Answer the patient's question using the research documents provided below. Be warm and professional — like a thoughtful doctor explaining results.

RULES (follow strictly):

GROUNDING (for faithfulness):
- Base your ENTIRE response on the documents below. Every factual statement you make must come from a specific document — cite it with [1], [2], etc.
- Prefer the documents' own wording when describing findings. Do not paraphrase in ways that change meaning or add nuance not present in the source.
- Do NOT add medical facts, background knowledge, general explanations, or conclusions that are not explicitly stated in the documents.
- Connective phrases for readability (e.g., "According to this research...") are allowed but must NOT introduce new factual claims.

RELEVANCE (for answering the question):
- Answer exactly what the question asks. Start by directly addressing the question in your opening sentence.
- Include ONLY information from the documents that directly helps answer the question. Do not include tangential findings from the same documents.
- If the patient context is provided, you may note how a document's findings relate to it, but ONLY if the document explicitly discusses matching values or conditions.

WHEN NO ANSWER EXISTS:
- If the documents contain genuinely NO information relevant to the question, respond with: "I wasn't able to find information about that in the available research documents." and set used_citations to [].
- But if the documents DO contain relevant information — including study details, authors, locations, or methodology — use it to answer.

CITATIONS:
- Cite every factual claim with [1], [2], etc.
- In used_citations, list ONLY the citation numbers you actually referenced.

PATIENT CONTEXT:
{health_context}

DOCUMENTS:
{docs_text}

"""

    # Add conversation history if exists
    if conversation_history:
        system_prompt += f"\nPREVIOUS CONVERSATION:\n{conversation_history}\n"

    system_prompt += f"\nCURRENT QUESTION:\n{current_query}\n\nAnswer the question directly, citing every claim from the documents:"

    logger.info("Generating structured response with Azure OpenAI (LangChain)")

    # Generate structured response
    result: MedicalResponse = structured_llm.invoke([HumanMessage(content=system_prompt)])

    logger.info(f"Response generated successfully. Used citations: {result.used_citations}")

    # Return as assistant message + used citations
    return {
        "messages": [AIMessage(content=result.response)],
        "used_citations": result.used_citations
    }
