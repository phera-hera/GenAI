"""
Medical RAG retrieval using LlamaIndex CitationQueryEngine.

Provides direct retrieval with inline citations from medical research papers.
"""

import logging
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import BaseModel, Field

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Response Model
# ============================================================================


class MedicalAnalysisResponse(BaseModel):
    """Structured output from medical RAG system."""

    agent_reply: str = Field(
        ...,
        description="Complete analysis including summary, detailed content, and personalized insights based on research",
    )
    disclaimers: str = Field(
        ...,
        description="Medical disclaimer stating this is informational only, not medical advice",
    )


# ============================================================================
# Citation Registry
# ============================================================================


class CitationRegistry:
    """Registry to capture and extract unique citations from query responses."""

    def __init__(self):
        self.source_nodes = []

    def add_sources(self, response):
        """Add source nodes from a query engine response."""
        if hasattr(response, "source_nodes"):
            self.source_nodes.extend(response.source_nodes)

    def get_unique_citations(self) -> list[dict[str, Any]]:
        """Extract unique citations with metadata."""
        seen_paper_ids = set()
        unique_citations = []

        for source in self.source_nodes:
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


# ============================================================================
# RAG Query Engine Builder
# ============================================================================


def build_citation_query_engine() -> CitationQueryEngine:
    """
    Build a CitationQueryEngine with vector store connection.

    Returns:
        Configured CitationQueryEngine
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

    return citation_engine


# ============================================================================
# Helper Functions
# ============================================================================


def _format_health_profile(profile: dict[str, Any]) -> str:
    """Format health profile for inclusion in query."""
    parts = []

    if age := profile.get("age"):
        parts.append(f"Age: {age}")

    if symptoms := profile.get("symptoms"):
        parts.append(f"Symptoms: {', '.join(symptoms)}")

    if diagnoses := profile.get("diagnoses"):
        parts.append(f"Diagnoses: {', '.join(diagnoses)}")

    if menstrual := profile.get("menstrual_status"):
        parts.append(f"Menstrual status: {menstrual}")

    if birth_control := profile.get("birth_control"):
        parts.append(f"Birth control: {', '.join(birth_control)}")

    if hrt := profile.get("hormone_therapy"):
        parts.append(f"Hormone therapy: {', '.join(hrt)}")

    if ethnicity := profile.get("ethnicity"):
        parts.append(f"Ethnicity: {', '.join(ethnicity)}")

    return "\n".join(parts) if parts else ""


def _build_query_prompt(ph_value: float, health_profile: dict[str, Any] | None = None) -> str:
    """
    Build a comprehensive query prompt for the RAG system.

    Args:
        ph_value: The pH measurement
        health_profile: Optional user health context

    Returns:
        Formatted query prompt
    """
    # Base query
    query_parts = [
        f"Analyze vaginal pH reading of {ph_value}.",
        "",
        "Provide a comprehensive medical analysis that includes:",
        "1. What this pH level indicates",
        "2. Possible causes or conditions associated with this pH level",
        "3. Relevant research findings and evidence",
    ]

    # Add health profile context if available
    if health_profile:
        profile_str = _format_health_profile(health_profile)
        if profile_str:
            query_parts.extend([
                "",
                "User Health Context:",
                profile_str,
                "",
                "4. Personalized insights considering the user's specific health profile",
            ])

    query_parts.extend([
        "",
        "Important:",
        "- Use ONLY information from medical research papers",
        "- Cite specific findings with [1], [2] references",
        "- pH 3.8-4.5: healthy acidic range",
        "- pH 4.5-5.0: slightly elevated",
        "- pH 5.0-6.0: may indicate bacterial imbalance",
        "- pH >6.0 or <3.8: needs medical attention",
    ])

    return "\n".join(query_parts)


# ============================================================================
# Query Function
# ============================================================================


async def query_medical_rag(
    ph_value: float,
    health_profile: dict[str, Any] | None = None,
) -> tuple[MedicalAnalysisResponse, list[dict[str, Any]]]:
    """
    Query the medical RAG system with a pH reading.

    Args:
        ph_value: The pH measurement
        health_profile: Optional user health context

    Returns:
        Tuple of (structured medical analysis, list of citation dicts)
    """
    logger.info(f"Querying medical RAG: pH={ph_value}")

    # Create citation registry
    citation_registry = CitationRegistry()

    # Build query engine
    citation_engine = build_citation_query_engine()

    # Build comprehensive query with health context
    query = _build_query_prompt(ph_value, health_profile)

    try:
        # Execute RAG query
        logger.info("Executing RAG retrieval...")
        response = await citation_engine.aquery(query)

        # Capture citations
        citation_registry.add_sources(response)

        logger.info(f"Retrieved response with {len(citation_registry.source_nodes)} source nodes")

        # Extract unique citations
        citations = citation_registry.get_unique_citations()
        logger.info(f"Extracted {len(citations)} unique citations")

        # Format response with structured output
        structured_llm = AzureOpenAI(
            engine=settings.azure_openai_deployment_name,
            model=settings.azure_openai_deployment_name,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=0.0,  # Deterministic formatting
        ).as_structured_llm(output_cls=MedicalAnalysisResponse)

        formatting_prompt = f"""Convert the following medical analysis into the required structured format.

Retrieved Analysis:
{str(response)}

Instructions:
- Place the complete analysis in the agent_reply field
- Include all cited references [1], [2], etc. in the response
- ALWAYS include this exact medical disclaimer in the disclaimers field:
  "This analysis is for informational purposes only and does NOT constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns. If you experience severe symptoms, seek immediate medical attention."

Return the structured response."""

        structured_response = await structured_llm.acomplete(formatting_prompt)
        logger.info("Structured output generated successfully")

        return structured_response.raw, citations

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise
