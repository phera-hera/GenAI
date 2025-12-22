"""
Query Engine for Medical RAG

Provides a high-level query interface that combines:
- Semantic retrieval from medical papers
- Context synthesis and response generation
- Citation tracking and source attribution
- Medical safety guardrails

This is the main entry point for querying the medical knowledge base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.core.config import settings
from app.db.models import ChunkType
from rag.retriever import (
    MedicalPaperRetriever,
    MultiQueryRetriever,
    RetrievalConfig,
    RetrievalResult,
    RerankStrategy,
    create_retriever,
)

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates for Medical RAG
# =============================================================================

MEDICAL_QA_TEMPLATE = PromptTemplate(
    """\
You are a medical research assistant specializing in women's vaginal health.
Your role is to synthesize information from peer-reviewed research papers to provide
evidence-based health insights.

IMPORTANT GUIDELINES:
- Base your response ONLY on the provided research excerpts
- Always cite your sources using the paper information provided
- Never prescribe medications or make specific diagnoses
- If the research is inconclusive or conflicting, acknowledge this
- Use clear, accessible language while maintaining medical accuracy
- If the query is outside vaginal health scope, politely redirect

CONTEXT FROM RESEARCH PAPERS:
{context_str}

USER QUERY: {query_str}

Provide a clear, evidence-based response that:
1. Directly addresses the user's question
2. Cites specific research findings with paper references
3. Notes any limitations or areas where more research is needed
4. Includes relevant caveats about individual variation

RESPONSE:
"""
)

REFINE_TEMPLATE = PromptTemplate(
    """\
You are refining an existing answer about women's vaginal health based on additional research.

ORIGINAL QUERY: {query_str}

EXISTING ANSWER:
{existing_answer}

ADDITIONAL RESEARCH CONTEXT:
{context_msg}

Given the new research context, refine the original answer to be more complete and accurate.
Integrate any new relevant information while maintaining proper citations.
If the new context contradicts the previous answer, acknowledge and explain the discrepancy.

REFINED ANSWER:
"""
)

SUMMARY_TEMPLATE = PromptTemplate(
    """\
You are summarizing medical research findings about women's vaginal health.

Based on the following research excerpts, provide a comprehensive summary that addresses: {query_str}

RESEARCH EXCERPTS:
{context_str}

Create a well-organized summary that:
1. Synthesizes key findings across papers
2. Notes areas of consensus and disagreement
3. Highlights limitations of the research
4. Provides proper citations

SUMMARY:
"""
)


@dataclass
class QueryResult:
    """Result from a query operation."""

    response: str
    source_nodes: list[NodeWithScore]
    query_text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0

    @property
    def has_sources(self) -> bool:
        return len(self.source_nodes) > 0

    @property
    def num_sources(self) -> int:
        return len(self.source_nodes)

    def get_source_texts(self) -> list[str]:
        """Get the text content of all source nodes."""
        return [node.node.get_content() for node in self.source_nodes]

    def format_citations(self) -> str:
        """Format citations as a readable string."""
        if not self.citations:
            return ""

        lines = ["Sources:"]
        for i, citation in enumerate(self.citations, 1):
            title = citation.get("paper_title", "Unknown")
            authors = citation.get("paper_authors", "Unknown authors")
            doi = citation.get("paper_doi", "")

            if authors and "," in authors:
                authors = authors.split(",")[0].strip() + " et al."

            citation_line = f"[{i}] {authors}. \"{title}\""
            if doi:
                citation_line += f" DOI: {doi}"
            lines.append(citation_line)

        return "\n".join(lines)


class MedicalQueryEngine:
    """
    High-level query engine for medical paper retrieval and synthesis.

    Combines retrieval, response synthesis, and citation tracking
    into a simple query interface.

    Args:
        retriever: MedicalPaperRetriever or MultiQueryRetriever
        llm: Optional LLM (uses Azure OpenAI if not provided)
        response_mode: How to synthesize responses
        use_multi_query: Whether to use multi-query retrieval
    """

    def __init__(
        self,
        retriever: MedicalPaperRetriever | MultiQueryRetriever | None = None,
        llm: LLM | None = None,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        use_multi_query: bool = False,
    ):
        self._retriever = retriever
        self._llm = llm
        self.response_mode = response_mode
        self.use_multi_query = use_multi_query
        self._query_engine: RetrieverQueryEngine | None = None

    @property
    def llm(self) -> LLM:
        """Get or create the LLM."""
        if self._llm is None:
            from app.services.azure_openai import get_llama_index_llm

            self._llm = get_llama_index_llm()
        return self._llm

    @property
    def retriever(self) -> MedicalPaperRetriever | MultiQueryRetriever:
        """Get or create the retriever."""
        if self._retriever is None:
            if self.use_multi_query:
                from rag.retriever import create_multi_query_retriever

                self._retriever = create_multi_query_retriever()
            else:
                self._retriever = create_retriever()
        return self._retriever

    def _get_response_synthesizer(self):
        """Create a response synthesizer with medical prompts."""
        return get_response_synthesizer(
            llm=self.llm,
            response_mode=self.response_mode,
            text_qa_template=MEDICAL_QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
            summary_template=SUMMARY_TEMPLATE,
        )

    async def aquery(self, query: str, **kwargs: Any) -> QueryResult:
        """
        Execute an async query against the medical knowledge base.

        Args:
            query: User's question or search query
            **kwargs: Additional query parameters

        Returns:
            QueryResult with response, sources, and citations
        """
        import time

        start_time = time.time()

        logger.info(f"Processing query: {query[:100]}...")

        try:
            # Retrieve relevant chunks
            if isinstance(self.retriever, MultiQueryRetriever):
                retrieval_result = await self.retriever.retrieve(query)
                source_nodes = retrieval_result.nodes
            else:
                query_bundle = QueryBundle(query_str=query)
                source_nodes = await self.retriever._aretrieve(query_bundle)
                retrieval_result = RetrievalResult(
                    nodes=source_nodes,
                    query_text=query,
                )

            # Generate response using LlamaIndex
            if source_nodes:
                response_synthesizer = self._get_response_synthesizer()
                response = await response_synthesizer.asynthesize(
                    query=query,
                    nodes=source_nodes,
                )
                response_text = response.response
            else:
                response_text = (
                    "I couldn't find relevant information in the medical research "
                    "database to answer your question. Please try rephrasing your "
                    "query or ask about a topic related to women's vaginal health."
                )

            # Extract citations
            citations = retrieval_result.get_citations()

            elapsed_ms = int((time.time() - start_time) * 1000)

            result = QueryResult(
                response=response_text,
                source_nodes=source_nodes,
                query_text=query,
                citations=citations,
                processing_time_ms=elapsed_ms,
                metadata={
                    "response_mode": self.response_mode.value,
                    "use_multi_query": self.use_multi_query,
                    "num_sources": len(source_nodes),
                },
            )

            logger.info(
                f"Query completed in {elapsed_ms}ms with {len(source_nodes)} sources"
            )

            return result

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def query(self, query: str, **kwargs: Any) -> QueryResult:
        """
        Synchronous query wrapper.

        For async contexts, use aquery() instead.
        """
        import asyncio

        return asyncio.run(self.aquery(query, **kwargs))


class PHAnalysisEngine:
    """
    Specialized query engine for pH value analysis.

    Combines pH readings with health profile data to generate
    personalized, evidence-based health insights.
    """

    def __init__(
        self,
        query_engine: MedicalQueryEngine | None = None,
    ):
        self.query_engine = query_engine or MedicalQueryEngine()

    def _build_ph_query(
        self,
        ph_value: float,
        symptoms: list[str] | None = None,
        health_context: dict[str, Any] | None = None,
    ) -> str:
        """Build a comprehensive query based on pH and health data."""
        parts = [f"vaginal pH level of {ph_value}"]

        if symptoms:
            symptom_str = ", ".join(symptoms)
            parts.append(f"symptoms including {symptom_str}")

        if health_context:
            age = health_context.get("age")
            if age:
                parts.append(f"age {age}")

            conditions = health_context.get("medical_history", {}).get("conditions", [])
            if conditions:
                parts.append(f"history of {', '.join(conditions)}")

        query = (
            f"What does a {' with '.join(parts)} indicate about vaginal health? "
            "What are the potential causes and recommended next steps based on research?"
        )

        return query

    async def analyze_ph(
        self,
        ph_value: float,
        symptoms: list[str] | None = None,
        health_context: dict[str, Any] | None = None,
    ) -> QueryResult:
        """
        Analyze a pH reading with health context.

        Args:
            ph_value: Measured vaginal pH value
            symptoms: List of current symptoms
            health_context: Additional health information

        Returns:
            QueryResult with personalized analysis
        """
        query = self._build_ph_query(ph_value, symptoms, health_context)

        result = await self.query_engine.aquery(query)

        # Add pH analysis metadata
        result.metadata["ph_value"] = ph_value
        result.metadata["symptoms"] = symptoms or []
        result.metadata["risk_level"] = self._assess_risk_level(ph_value, symptoms)

        return result

    def _assess_risk_level(
        self,
        ph_value: float,
        symptoms: list[str] | None,
    ) -> str:
        """
        Determine risk level based on pH and symptoms.

        Risk levels:
        - NORMAL: pH 3.8-4.5, no symptoms
        - MONITOR: pH 3.8-4.5 with mild symptoms
        - CONCERNING: pH 4.5-5.0
        - URGENT: pH > 5.0
        """
        has_symptoms = bool(symptoms and len(symptoms) > 0)

        if ph_value >= settings.ph_concerning_threshold:
            return "URGENT"
        elif ph_value > settings.ph_normal_max:
            return "CONCERNING"
        elif settings.ph_normal_min <= ph_value <= settings.ph_normal_max:
            return "MONITOR" if has_symptoms else "NORMAL"
        else:
            # pH below normal range (too acidic)
            return "MONITOR"


# =============================================================================
# Factory Functions
# =============================================================================


def create_query_engine(
    top_k: int | None = None,
    chunk_types: list[ChunkType] | None = None,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    use_multi_query: bool = False,
    **kwargs: Any,
) -> MedicalQueryEngine:
    """
    Create a configured MedicalQueryEngine.

    Args:
        top_k: Number of chunks to retrieve
        chunk_types: Filter by specific chunk types
        response_mode: How to synthesize responses
        use_multi_query: Whether to use multi-query retrieval
        **kwargs: Additional retriever configuration

    Returns:
        Configured query engine
    """
    retriever = create_retriever(
        top_k=top_k or settings.vector_similarity_top_k,
        chunk_types=chunk_types,
        **kwargs,
    )

    return MedicalQueryEngine(
        retriever=retriever,
        response_mode=response_mode,
        use_multi_query=use_multi_query,
    )


def create_ph_analyzer(
    top_k: int | None = None,
    **kwargs: Any,
) -> PHAnalysisEngine:
    """
    Create a PHAnalysisEngine for pH-based queries.

    Args:
        top_k: Number of chunks to retrieve
        **kwargs: Additional configuration

    Returns:
        Configured pH analyzer
    """
    query_engine = create_query_engine(
        top_k=top_k,
        use_multi_query=True,  # Better recall for pH-related queries
        **kwargs,
    )
    return PHAnalysisEngine(query_engine=query_engine)

