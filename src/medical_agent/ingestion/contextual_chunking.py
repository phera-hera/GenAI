"""
Contextual chunking: prepend document context to each chunk.

Based on Anthropic's contextual retrieval research (2024):
- Generates a short context for each chunk using the full document
- Prepends context to chunk text before embedding
- Reduces retrieval errors by 49% when combined with hybrid search
- Improves retrieval accuracy by up to 67% in Anthropic's benchmarks

The problem this solves:
    Traditional chunking destroys context. A chunk saying "The treatment group
    showed 85% improvement" doesn't tell the retriever what treatment, what
    condition, or what study. Contextual headers provide that missing context.

Example:
    Original chunk: "The treatment group showed 85% improvement (p<0.001)."

    With context: "This chunk discusses findings from a 2023 randomized controlled
    trial on metronidazole treatment for bacterial vaginosis in women with PCOS.

    The treatment group showed 85% improvement (p<0.001)."
"""

import logging
from llama_index.core.schema import BaseNode, TextNode
from langchain_openai import AzureChatOpenAI
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{doc_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context (2-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.

The context should:
1. Identify the study/paper (title, authors, year if available)
2. Briefly state what the chunk is about (e.g., "methodology", "results for X treatment", "discussion of Y factor")
3. Include key medical terms or conditions relevant to the chunk

Answer only with the succinct context and nothing else."""


async def add_contextual_headers(
    nodes: list[BaseNode],
    full_document_text: str,
) -> list[BaseNode]:
    """
    Add document-level context to each chunk.

    Calls gpt-4o-mini once per chunk to generate a 2-3 sentence
    context prefix that situates the chunk within the document.

    Args:
        nodes: Chunk nodes from parser
        full_document_text: Complete document text (will be truncated to first 8000 chars for context window)

    Returns:
        Nodes with contextual headers prepended to text
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,  # Deterministic for consistent context generation
    )

    # Truncate document for context window (8000 chars ≈ 2000 tokens)
    doc_context = full_document_text[:8000]

    result_nodes = []
    success_count = 0
    failure_count = 0

    for idx, node in enumerate(nodes):
        chunk_text = node.get_content()

        # Skip very short chunks (< 20 words) - not worth the LLM call
        if len(chunk_text.split()) < 20:
            result_nodes.append(node)
            continue

        try:
            # Generate contextual header
            response = await llm.ainvoke(
                CONTEXT_PROMPT.format(doc_text=doc_context, chunk_text=chunk_text[:1000])
            )
            context = response.content.strip()

            # Prepend context to chunk text
            contextualized_text = f"{context}\n\n{chunk_text}"

            # Create new node with contextualized text
            new_node = TextNode(
                text=contextualized_text,
                metadata={
                    **node.metadata,
                    "contextual_header": context,
                    "has_context": True,
                },
                id_=node.node_id,
            )
            result_nodes.append(new_node)
            success_count += 1

            if (idx + 1) % 10 == 0:
                logger.info(f"Contextual headers: {idx + 1}/{len(nodes)} chunks processed")

        except Exception as e:
            logger.warning(f"Contextual header generation failed for chunk {idx + 1}: {e}")
            result_nodes.append(node)
            failure_count += 1

    logger.info(
        f"Contextual chunking complete: {success_count} chunks contextualized, "
        f"{failure_count} failures, {len(nodes) - success_count - failure_count} skipped"
    )

    return result_nodes
