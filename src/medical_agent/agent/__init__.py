"""
Medical ReActAgent using LlamaIndex native components.

Replaces the custom LangGraph workflow with a simple ReActAgent.

Usage:
    from medical_agent.agent import query_medical_agent

    response = await query_medical_agent(
        ph_value=4.8,
        health_profile={"age": 28, "symptoms": ["mild discharge"]},
    )
    print(response.summary)
"""

from medical_agent.agent.react_agent import (
    MedicalAnalysisResponse,
    build_medical_agent,
    query_medical_agent,
)

__all__ = [
    "build_medical_agent",
    "query_medical_agent",
    "MedicalAnalysisResponse",
]

