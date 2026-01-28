"""
LangGraph workflow definition for medical RAG agent.

Builds a simple linear graph: retrieve → generate → END
Uses MemorySaver for multi-turn conversation memory.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from medical_agent.agents.nodes import generate_node, retrieve_node
from medical_agent.agents.state import MedicalAgentState

logger = logging.getLogger(__name__)


def build_medical_rag_graph():
    """
    Build and compile the medical RAG agent graph.

    Graph flow:
        START → retrieve_node → generate_node → END

    Features:
        - MemorySaver for session-based conversation history
        - Simple linear flow (no conditional routing yet)
        - Pure function nodes (no agents/tools)

    Returns:
        Compiled LangGraph application with checkpointer
    """
    logger.info("Building medical RAG graph")

    # Create state graph
    workflow = StateGraph(MedicalAgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Define edges (linear flow)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Add memory for multi-turn conversations
    memory = MemorySaver()

    # Compile graph
    app = workflow.compile(checkpointer=memory)

    logger.info("Medical RAG graph compiled successfully")

    return app


# Global compiled graph instance
medical_rag_app = build_medical_rag_graph()
