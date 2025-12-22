"""
LangGraph agent module for medical reasoning workflow.

This module provides the complete agent workflow for processing medical queries:
- Query Analyzer: Parse pH, extract symptoms, generate search queries
- Retriever: Vector search for relevant research chunks
- Risk Assessor: Evaluate pH against normal range, determine urgency
- Reasoner: Analyze evidence, synthesize insights from research
- Response Generator: Format response with citations and disclaimers

Key Components:
- prompts: System prompts, response templates, guardrails, and disclaimers
- nodes: Individual agent node implementations
- state: Agent state schema and data classes
- graph: LangGraph workflow definition and execution

Usage:
    from agent.graph import run_medical_agent

    result = await run_medical_agent(
        ph_value=4.8,
        health_profile={"age": 28, "symptoms": ["mild discharge"]},
    )
    print(result["final_response"]["full_response"])
"""

from agent import nodes, prompts
from agent.graph import (
    get_medical_agent_graph,
    run_medical_agent,
    run_medical_agent_sync,
    get_graph_mermaid,
)
from agent.state import (
    AgentState,
    Citation,
    ConflictingEvidence,
    EvidenceSummary,
    FinalResponse,
    HealthProfile,
    PHCategory,
    QueryAnalysis,
    ReasoningOutput,
    RetrievedChunk,
    RiskAssessment,
    RiskLevel,
    create_initial_state,
)

__all__ = [
    # Submodules
    "nodes",
    "prompts",
    # Graph functions
    "get_medical_agent_graph",
    "run_medical_agent",
    "run_medical_agent_sync",
    "get_graph_mermaid",
    # State classes
    "AgentState",
    "Citation",
    "ConflictingEvidence",
    "EvidenceSummary",
    "FinalResponse",
    "HealthProfile",
    "PHCategory",
    "QueryAnalysis",
    "ReasoningOutput",
    "RetrievedChunk",
    "RiskAssessment",
    "RiskLevel",
    "create_initial_state",
]

