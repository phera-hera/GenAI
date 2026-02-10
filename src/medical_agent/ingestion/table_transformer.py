"""
Convert table chunks to natural language for better embedding search.

Medical RAG systems achieve 90% accuracy with proper table-to-NL transformation.
Tables with columns like "Treatment | n | Cure Rate | p-value" are hard to search
via embeddings. Converting to natural language makes them semantically searchable.

Example:
    Table: "Treatment | Cure Rate\nMetronidazole | 85%\nClindamycin | 78%"
    →
    Natural Language: "Metronidazole showed an 85% cure rate for bacterial vaginosis
    treatment, while Clindamycin achieved a 78% cure rate."
"""

import logging
from llama_index.core.schema import BaseNode, TextNode
from langchain_openai import AzureChatOpenAI
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

TABLE_PROMPT = """Convert this research table data into clear, factual natural language sentences.

**Instructions:**
1. Write 2-5 standalone factual sentences that capture the key data points
2. Each sentence should be searchable and answer potential research questions
3. Preserve ALL numbers, percentages, p-values, and statistical data EXACTLY as shown
4. Include context (what the table is about) in each sentence
5. Use proper medical terminology
6. Make data relationships explicit (e.g., "compared to", "higher than", "associated with")

**Table data:**
{table_text}

**Natural language summary:**"""


async def transform_table_chunks(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Find table chunks and convert them to natural language.

    Replaces table text with LLM-generated natural language while
    preserving original table data in metadata for reference.

    Args:
        nodes: All chunk nodes from parser

    Returns:
        Nodes with table chunks transformed to natural language
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,  # gpt-4o (not mini)
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,  # Deterministic for factual extraction
    )

    result_nodes = []
    table_count = 0

    for node in nodes:
        # Check if this is a table chunk
        chunk_type = (node.metadata.get("chunk_type", "") or "").lower()
        doc_items = node.metadata.get("doc_items", [])
        is_table = chunk_type == "table" or (
            doc_items and doc_items[0].get("type") == "table"
        )

        if is_table:
            table_text = node.get_content()

            # Skip empty or very short tables
            if not table_text.strip() or len(table_text.strip()) < 20:
                logger.warning(f"Skipping empty/short table chunk")
                result_nodes.append(node)
                continue

            try:
                logger.info(f"Transforming table chunk: {table_text[:80]}...")
                response = await llm.ainvoke(TABLE_PROMPT.format(table_text=table_text))
                natural_text = response.content.strip()

                # Create new node with natural language text
                new_node = TextNode(
                    text=natural_text,
                    metadata={
                        **node.metadata,
                        "original_table_text": table_text[:500],  # Keep reference (truncated)
                        "table_transformed": True,
                    },
                    id_=node.node_id,
                )
                result_nodes.append(new_node)
                table_count += 1
                logger.info(f"✓ Table transformed: {natural_text[:100]}...")

            except Exception as e:
                logger.warning(f"Table transform failed, keeping original: {e}")
                result_nodes.append(node)
        else:
            result_nodes.append(node)

    if table_count > 0:
        logger.info(f"Table transform complete: {table_count} tables → natural language")

    return result_nodes
