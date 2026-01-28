"""
LangGraph node functions for medical RAG workflow.

Each node is a pure function that takes state and returns state updates.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from medical_agent.agents.llamaindex_retrieval import retrieve_nodes
from medical_agent.agents.state import MedicalAgentState
from medical_agent.agents.utils import build_health_context, format_retrieved_nodes
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


def retrieve_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Retrieval node: Fetches relevant medical research chunks.

    Process:
        1. Extract user's latest message
        2. Build search query from pH + health context
        3. Retrieve nodes using LlamaIndex retriever
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

    # Build enhanced query with health context
    ph_value = state.get("ph_value", 0.0)
    health_profile = state.get("health_profile", {})
    health_context = build_health_context(ph_value, health_profile)

    # Combine query with context for better retrieval
    enhanced_query = f"{user_query}\n\nHealth Context:\n{health_context}"

    logger.info(f"Retrieving nodes for query: {user_query[:100]}...")

    # Retrieve nodes using existing LlamaIndex retriever
    nodes = retrieve_nodes(query=enhanced_query, similarity_top_k=5)

    # Format nodes into citation text
    docs_text, citations = format_retrieved_nodes(nodes)

    logger.info(f"Retrieved {len(citations)} citations")

    return {
        "docs_text": docs_text,
        "citations": citations
    }


def generate_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Generation node: Produces medical response with inline citations.

    Process:
        1. Get LLM from Azure OpenAI (LangChain)
        2. Build strict medical prompt with docs_text
        3. Include conversation history for context
        4. Generate response with inline [1][2] citations
        5. Return assistant message

    Args:
        state: Current agent state with docs_text and conversation history

    Returns:
        State update with new assistant message
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

    # Build strict medical prompt
    system_prompt = f"""You are a medical research assistant specialized in women's reproductive health and vaginal pH analysis.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided medical documents below
2. Cite sources inline using the citation markers [1][2] that appear in the documents
3. If the documents do not contain relevant information to answer the question, respond EXACTLY with:
   "No relevant medical research found in the available documents."
4. DO NOT use external knowledge or make assumptions beyond what's in the documents
5. Be concise and medically accurate
6. Focus on evidence-based information from peer-reviewed research

PATIENT CONTEXT:
{health_context}

MEDICAL DOCUMENTS:
{docs_text}

"""

    # Add conversation history if exists
    if conversation_history:
        system_prompt += f"\nPREVIOUS CONVERSATION:\n{conversation_history}\n"

    system_prompt += f"\nCURRENT QUESTION:\n{current_query}\n\nProvide a clear, evidence-based answer with inline citations:"

    logger.info("Generating response with Azure OpenAI (LangChain)")

    # Generate response using LangChain
    response = llm.invoke([HumanMessage(content=system_prompt)])

    logger.info("Response generated successfully")

    # Return as assistant message (LangChain returns AIMessage object)
    return {
        "messages": [response]  # LangChain message format - native interoperability
    }
