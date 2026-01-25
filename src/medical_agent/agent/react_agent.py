"""
Medical ReActAgent using LlamaIndex native components.

ReActAgent
that uses VectorStoreIndex for retrieval.
"""

import logging
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.workflow import Context
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import BaseModel, Field

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Response Model (matches API QueryResponse structure)
# ============================================================================


class MedicalAnalysisResponse(BaseModel):
    """Structured output from medical agent."""

    risk_level: str = Field(
        ...,
        description="Risk assessment: NORMAL, MONITOR, CONCERNING, or URGENT",
    )
    summary: str = Field(..., description="Brief 1-2 sentence summary")
    main_content: str = Field(
        ...,
        description="Detailed analysis (2-3 paragraphs) based on research",
    )
    personalized_insights: list[str] = Field(
        default_factory=list,
        description="3-5 bullet points specific to user's profile",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="3-5 actionable recommendations",
    )
    disclaimers: str = Field(
        ...,
        description="Medical disclaimer stating this is informational only, not medical advice",
    )


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are a medical research assistant specializing in vaginal pH analysis and women's health.

## Your Role
Analyze pH readings in context of the user's health profile and provide evidence-based insights from medical research.

## Guidelines
1. **Risk Assessment**: Classify pH readings as:
   - NORMAL: pH 3.8-4.5 (healthy acidic range)
   - MONITOR: pH 4.5-5.0 (slightly elevated, watch for symptoms)
   - CONCERNING: pH 5.0-6.0 (may indicate bacterial imbalance)
   - URGENT: pH >6.0 or <3.8 (needs medical attention)

2. **Research-Based**: Use the medical_research tool to find relevant studies. Cite specific findings.

3. **Personalization**: Consider user's:
   - Age and menstrual status
   - Symptoms (discharge, odor, irritation)
   - Birth control and hormone therapy
   - Diagnoses (PCOS, endometriosis, etc.)
   - Ethnicity (some conditions vary by ethnicity)

4. **Be Specific**: Reference actual study findings, not generic advice.

5. **Next Steps**: Provide actionable recommendations based on findings.

## CRITICAL: Medical Disclaimers
ALWAYS include this exact disclaimer in your response:
"This analysis is for informational purposes only and does NOT constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns. If you experience severe symptoms, seek immediate medical attention."

## Output Format
Respond with structured data including risk_level, summary, main_content, personalized_insights, next_steps, and disclaimers.
"""


# ============================================================================
# Agent Builder
# ============================================================================


class CitationRegistry:
    """Registry to capture citations from all tool calls during agent execution."""

    def __init__(self):
        self.citations = []

    def add(self, response):
        """Add source nodes from a query engine response."""
        if hasattr(response, "source_nodes"):
            self.citations.extend(response.source_nodes)

    def get_unique_citations(self) -> list[dict[str, Any]]:
        """Extract unique citations with metadata."""
        seen_paper_ids = set()
        unique_citations = []

        for source in self.citations:
            if hasattr(source, "node") and source.node.metadata:
                metadata = source.node.metadata
                paper_id = metadata.get("paper_id", "")

                # Avoid duplicates
                if paper_id and paper_id not in seen_paper_ids:
                    seen_paper_ids.add(paper_id)
                    unique_citations.append({
                        "paper_id": paper_id,
                        "title": metadata.get("title"),
                        "authors": metadata.get("authors"),
                        "doi": metadata.get("doi"),
                        "relevant_section": source.node.text[:200] if hasattr(source.node, "text") else None,
                    })

        return unique_citations


def build_medical_agent(
    health_profile: dict[str, Any] | None = None,
    citation_registry: CitationRegistry | None = None,
) -> ReActAgent:
    """
    Build a ReActAgent with medical research retrieval tool.

    Args:
        health_profile: Optional user health context

    Returns:
        Configured ReActAgent
    """
    # Initialize Azure OpenAI LLM
    llm = AzureOpenAI(
        engine=settings.azure_openai_deployment_name,
        model=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.1,  # Low temperature for medical accuracy
    )

    # Initialize embeddings
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

    # Connect to vector store
    vector_store = PGVectorStore.from_params(
        database=settings.postgres_db,
        host=settings.postgres_host,
        password=settings.postgres_password,
        port=settings.postgres_port,
        user=settings.postgres_user,
        table_name="paper_chunks",  # LlamaIndex adds "data_" prefix
        embed_dim=settings.embedding_dimension,
        hybrid_search=True,
        text_search_config="english",
        perform_setup=False,
    )

    # Create VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Create CitationQueryEngine for in-line citations
    citation_engine = CitationQueryEngine.from_args(
        index=index,
        llm=llm,
        similarity_top_k=5,  # Retrieve top 5 chunks
        citation_chunk_size=512,  # Size of citation chunks
    )

    # Create citation-capturing wrapper function
    # This intercepts tool calls and captures source_nodes while returning text to agent
    def citation_tool_wrapper(query_str: str) -> str:
        """Wrapper that captures citations and returns text response."""
        response = citation_engine.query(query_str)

        # Capture citations if registry provided
        if citation_registry:
            citation_registry.add(response)

        # Return only text to agent (preserves ReAct reasoning flow)
        return str(response)

    # Wrap the function as a QueryEngineTool
    from llama_index.core.tools import FunctionTool

    research_tool = FunctionTool.from_defaults(
        fn=citation_tool_wrapper,
        name="medical_research",
        description=(
            "Search medical research papers about vaginal pH, bacterial vaginosis, "
            "menopause, hormones, and women's health. Use this to find evidence-based "
            "information about pH ranges, symptoms, diagnoses, and treatments. "
            "Query with specific medical terms for best results. "
            "Returns cited sources with [1], [2] references."
        ),
    )

    # Build system prompt with health profile context
    system_message = SYSTEM_PROMPT
    if health_profile:
        profile_context = _format_health_profile(health_profile)
        system_message += f"\n\n## User Health Profile\n{profile_context}"

    # Create ReActAgent with system prompt including health profile context
    agent = ReActAgent(
        tools=[research_tool],
        llm=llm,
        system_prompt=system_message,  # Include health profile context
        timeout=120.0,  # 2 minute timeout
        verbose=True,
    )

    return agent


def _format_health_profile(profile: dict[str, Any]) -> str:
    """Format health profile for system prompt."""
    parts = []

    if age := profile.get("age"):
        parts.append(f"- Age: {age}")

    if symptoms := profile.get("symptoms"):
        parts.append(f"- Symptoms: {', '.join(symptoms)}")

    if diagnoses := profile.get("diagnoses"):
        parts.append(f"- Diagnoses: {', '.join(diagnoses)}")

    if menstrual := profile.get("menstrual_status"):
        parts.append(f"- Menstrual status: {menstrual}")

    if birth_control := profile.get("birth_control"):
        parts.append(f"- Birth control: {', '.join(birth_control)}")

    if hrt := profile.get("hormone_therapy"):
        parts.append(f"- Hormone therapy: {', '.join(hrt)}")

    if ethnicity := profile.get("ethnicity"):
        parts.append(f"- Ethnicity: {', '.join(ethnicity)}")

    return "\n".join(parts) if parts else "No additional health context provided"


# ============================================================================
# Query Function
# ============================================================================


async def query_medical_agent(
    ph_value: float,
    health_profile: dict[str, Any] | None = None,
) -> tuple[MedicalAnalysisResponse, list[dict[str, Any]]]:
    """
    Query the medical agent with a pH reading.

    Args:
        ph_value: The pH measurement
        health_profile: Optional user health context

    Returns:
        Tuple of (structured medical analysis, list of citation dicts)
    """
    logger.info(f"Querying medical agent: pH={ph_value}")

    # Create citation registry to capture sources from all tool calls
    citation_registry = CitationRegistry()

    # Build agent with health context and citation registry
    agent = build_medical_agent(health_profile, citation_registry)

    # Construct query
    query = f"Analyze vaginal pH reading of {ph_value}. What does this indicate?"

    # Step 1: Run agent (reasoning + tool calls)
    try:
        # Create context for conversation state
        ctx = Context(agent)

        # Run agent (supports multi-step ReAct reasoning)
        logger.info("Starting agent run...")
        handler = agent.run(query, ctx=ctx)

        # Get final response
        agent_response = await handler
        logger.info(f"Agent reasoning complete: {str(agent_response)[:200]}...")

        # Extract citations from registry
        citations = citation_registry.get_unique_citations()
        logger.info(f"Extracted {len(citations)} citations from {len(citation_registry.citations)} total sources")

        # Step 2: Convert agent response to structured format
        # Use a separate LLM call with structured output to guarantee schema
        structured_llm = AzureOpenAI(
            engine=settings.azure_openai_deployment_name,
            model=settings.azure_openai_deployment_name,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=0.0,  # Deterministic formatting
        ).as_structured_llm(output_cls=MedicalAnalysisResponse)

        formatting_prompt = f"""Convert the following medical analysis into the required structured format.

Agent Analysis:
{str(agent_response)}

Instructions:
- Extract risk_level (must be exactly: NORMAL, MONITOR, CONCERNING, or URGENT)
- Create brief summary (1-2 sentences)
- Extract main_content (detailed analysis, 2-3 paragraphs)
- List personalized_insights as bullet points (3-5 items)
- List next_steps as actionable recommendations (3-5 items)
- ALWAYS include the medical disclaimer exactly as specified in the schema

Return the structured response."""

        structured_response = await structured_llm.acomplete(formatting_prompt)
        logger.info("Structured output generated successfully")

        return structured_response.raw, citations

    except Exception as e:
        logger.error(f"Agent query failed: {e}", exc_info=True)
        raise
