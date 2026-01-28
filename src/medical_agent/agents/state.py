"""
LangGraph state schema for medical RAG agent.

Defines the state structure that flows through the graph nodes.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class MedicalAgentState(TypedDict):
    """
    State schema for the medical RAG agent graph.

    Attributes:
        messages: Conversation history (managed by LangGraph's add_messages)
        ph_value: User's pH measurement
        health_profile: Dict containing age, symptoms, diagnoses, etc.
        docs_text: Formatted citation text from retrieval ("[1]: [Paper:page]: text")
        citations: List of citation metadata dicts
    """

    messages: Annotated[list, add_messages]  # Auto-managed conversation history
    ph_value: float
    health_profile: dict[str, Any]
    docs_text: str
    citations: list[dict[str, Any]]
