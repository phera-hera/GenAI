"""
LangGraph Workflow for Medical Reasoning Agent

Defines the complete agent workflow that orchestrates:
1. Query Analyzer - Parse pH, extract symptoms, generate search queries
2. Retriever - Vector search for relevant research chunks
3. Risk Assessor - Evaluate pH and symptoms for risk level
4. Reasoner - Analyze evidence and synthesize insights
5. Response Generator - Create formatted response with citations

The workflow is implemented as a LangGraph StateGraph with conditional
routing based on the analysis results.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes.query_analyzer import aquery_analyzer_node
from agent.nodes.reasoner import areasoner_node
from agent.nodes.response_generator import aresponse_generator_node
from agent.nodes.retriever import aretriever_node
from agent.nodes.risk_assessor import arisk_assessor_node
from agent.state import AgentState, HealthProfile, RiskLevel, create_initial_state
from app.core.config import settings
from app.services.langfuse_client import get_langfuse_client

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# =============================================================================
# Node Wrappers for Async Execution
# =============================================================================


async def query_analyzer(state: AgentState) -> AgentState:
    """Query Analyzer node wrapper."""
    logger.debug("Entering Query Analyzer node")
    return await aquery_analyzer_node(state)


async def retriever(state: AgentState) -> AgentState:
    """Retriever node wrapper."""
    logger.debug("Entering Retriever node")
    return await aretriever_node(state)


async def risk_assessor(state: AgentState) -> AgentState:
    """Risk Assessor node wrapper."""
    logger.debug("Entering Risk Assessor node")
    return await arisk_assessor_node(state)


async def reasoner(state: AgentState) -> AgentState:
    """Medical Reasoner node wrapper."""
    logger.debug("Entering Reasoner node")
    return await areasoner_node(state)


async def response_generator(state: AgentState) -> AgentState:
    """Response Generator node wrapper."""
    logger.debug("Entering Response Generator node")
    return await aresponse_generator_node(state)


# =============================================================================
# Conditional Routing Functions
# =============================================================================


def should_continue_after_retrieval(state: AgentState) -> Literal["risk_assessor", "response_generator"]:
    """
    Decide whether to continue to risk assessment or skip to response.

    If no chunks were retrieved, we may want to generate a response
    indicating that no relevant research was found.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Always continue to risk assessment - it can work without research
    # but may provide different recommendations
    return "risk_assessor"


def should_skip_reasoning(state: AgentState) -> Literal["reasoner", "response_generator"]:
    """
    Decide whether to skip the reasoning step.

    For URGENT cases, we may want to expedite the response.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    risk_assessment = state.get("risk_assessment", {})
    risk_level_str = risk_assessment.get("risk_level", "").upper()

    # For urgent cases with no research, skip directly to response
    retrieved_chunks = state.get("retrieved_chunks", [])

    if risk_level_str == "URGENT" and not retrieved_chunks:
        logger.info("Skipping reasoning for urgent case without research context")
        return "response_generator"

    return "reasoner"


# =============================================================================
# Graph Construction
# =============================================================================


def build_medical_agent_graph() -> StateGraph:
    """
    Build the LangGraph state graph for the medical reasoning agent.

    The workflow follows this structure:

    START -> query_analyzer -> retriever -> risk_assessor -> reasoner -> response_generator -> END

    With conditional routing:
    - After retrieval: Always continue to risk assessment
    - After risk assessment: May skip reasoning for urgent cases without context

    Returns:
        StateGraph ready for compilation
    """
    # Create the graph with AgentState as the state schema
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retriever", retriever)
    workflow.add_node("risk_assessor", risk_assessor)
    workflow.add_node("reasoner", reasoner)
    workflow.add_node("response_generator", response_generator)

    # Define edges
    # START -> query_analyzer
    workflow.add_edge(START, "query_analyzer")

    # query_analyzer -> retriever
    workflow.add_edge("query_analyzer", "retriever")

    # retriever -> risk_assessor (conditional, but always continues for now)
    workflow.add_conditional_edges(
        "retriever",
        should_continue_after_retrieval,
        {
            "risk_assessor": "risk_assessor",
            "response_generator": "response_generator",
        },
    )

    # risk_assessor -> reasoner or response_generator (conditional)
    workflow.add_conditional_edges(
        "risk_assessor",
        should_skip_reasoning,
        {
            "reasoner": "reasoner",
            "response_generator": "response_generator",
        },
    )

    # reasoner -> response_generator
    workflow.add_edge("reasoner", "response_generator")

    # response_generator -> END
    workflow.add_edge("response_generator", END)

    return workflow


def compile_medical_agent_graph() -> CompiledStateGraph:
    """
    Compile the medical agent graph for execution.

    Returns:
        Compiled StateGraph ready for invocation
    """
    workflow = build_medical_agent_graph()
    return workflow.compile()


# Global compiled graph instance
_compiled_graph: CompiledStateGraph | None = None


def get_medical_agent_graph(force_rebuild: bool = False) -> CompiledStateGraph:
    """
    Get or create the compiled medical agent graph.

    Args:
        force_rebuild: If True, rebuild the graph even if one exists

    Returns:
        Compiled StateGraph
    """
    global _compiled_graph

    if _compiled_graph is None or force_rebuild:
        logger.info("Building medical agent graph")
        _compiled_graph = compile_medical_agent_graph()
        logger.info("Medical agent graph compiled successfully")

    return _compiled_graph


# =============================================================================
# Agent Execution Interface
# =============================================================================


async def run_medical_agent(
    ph_value: float,
    health_profile: HealthProfile | dict[str, Any] | None = None,
    query_text: str | None = None,
    user_id: str | None = None,
    is_pregnant: bool = False,
    is_first_query: bool = False,
) -> dict[str, Any]:
    """
    Run the medical reasoning agent with the given inputs.

    This is the main entry point for executing the agent workflow.

    Args:
        ph_value: The pH value from the test strip
        health_profile: User's health profile
        query_text: Optional additional query text
        user_id: Optional user identifier
        is_pregnant: Whether the user is pregnant
        is_first_query: Whether this is the user's first query

    Returns:
        Dict containing the final response and metadata

    Example:
        >>> result = await run_medical_agent(
        ...     ph_value=4.8,
        ...     health_profile={"age": 28, "symptoms": ["mild discharge"]},
        ... )
        >>> print(result["final_response"]["full_response"])
    """
    logger.info(f"Starting medical agent for pH={ph_value}")
    start_time = time.time()

    # Create initial state
    state = create_initial_state(
        ph_value=ph_value,
        health_profile=health_profile,
        query_text=query_text,
        user_id=user_id,
        is_pregnant=is_pregnant,
        is_first_query=is_first_query,
    )

    # Set up Langfuse tracing
    langfuse = get_langfuse_client()
    if langfuse.is_configured():
        try:
            trace = langfuse.create_trace(
                name="medical_agent_run",
                user_id=user_id,
                session_id=state["session_id"],
                metadata={
                    "ph_value": ph_value,
                    "is_pregnant": is_pregnant,
                },
                tags=["medical_agent", "ph_analysis"],
            )
            if trace:
                state["langfuse_trace_id"] = trace.id
        except Exception as e:
            logger.warning(f"Failed to create Langfuse trace: {e}")

    # Get compiled graph
    graph = get_medical_agent_graph()

    # Run the workflow
    try:
        final_state = await graph.ainvoke(state)

        # Log completion to Langfuse
        if langfuse.is_configured() and state.get("langfuse_trace_id"):
            langfuse.log_event(
                trace=langfuse.client.trace(id=state["langfuse_trace_id"]),
                name="workflow_complete",
                metadata={
                    "processing_time_ms": final_state.get("processing_time_ms", 0),
                    "risk_level": final_state.get("risk_assessment", {}).get("risk_level"),
                    "errors_count": len(final_state.get("errors", [])),
                },
            )
            langfuse.flush()

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"Medical agent completed in {elapsed}ms")

        return {
            "session_id": final_state.get("session_id"),
            "ph_value": final_state.get("ph_value"),
            "risk_level": final_state.get("risk_assessment", {}).get("risk_level"),
            "final_response": final_state.get("final_response", {}),
            "citations": final_state.get("reasoning_output", {}).get("citations", []),
            "processing_time_ms": elapsed,
            "errors": final_state.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Medical agent execution failed: {e}")

        # Log error to Langfuse
        if langfuse.is_configured() and state.get("langfuse_trace_id"):
            langfuse.log_event(
                trace=langfuse.client.trace(id=state["langfuse_trace_id"]),
                name="workflow_error",
                level="ERROR",
                message=str(e),
            )
            langfuse.flush()

        raise


def run_medical_agent_sync(
    ph_value: float,
    health_profile: HealthProfile | dict[str, Any] | None = None,
    query_text: str | None = None,
    user_id: str | None = None,
    is_pregnant: bool = False,
    is_first_query: bool = False,
) -> dict[str, Any]:
    """
    Synchronous wrapper for run_medical_agent.

    For use in non-async contexts.

    Args:
        Same as run_medical_agent

    Returns:
        Same as run_medical_agent
    """
    import asyncio

    return asyncio.run(
        run_medical_agent(
            ph_value=ph_value,
            health_profile=health_profile,
            query_text=query_text,
            user_id=user_id,
            is_pregnant=is_pregnant,
            is_first_query=is_first_query,
        )
    )


# =============================================================================
# Graph Visualization (for debugging)
# =============================================================================


def get_graph_mermaid() -> str:
    """
    Get Mermaid diagram representation of the graph.

    Useful for documentation and debugging.

    Returns:
        Mermaid diagram string
    """
    graph = get_medical_agent_graph()
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Failed to generate Mermaid diagram: {e}")
        return """
stateDiagram-v2
    [*] --> query_analyzer
    query_analyzer --> retriever
    retriever --> risk_assessor
    risk_assessor --> reasoner
    reasoner --> response_generator
    response_generator --> [*]
"""


def print_graph() -> None:
    """Print the graph structure for debugging."""
    print(get_graph_mermaid())


