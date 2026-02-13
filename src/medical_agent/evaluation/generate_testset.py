"""
Synthetic test set generation from ingested paper chunks.

Fetches chunks from the database and uses RAGAS TestsetGenerator
to create synthetic question-answer pairs for evaluation.

Usage:
    python -m medical_agent.evaluation.generate_testset --size 20
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import text

from medical_agent.evaluation.ragas_config import (
    get_evaluator_embeddings,
    get_evaluator_llm,
    setup_langsmith_tracing,
)
from medical_agent.infrastructure.database.session import get_session_context

logger = logging.getLogger(__name__)

TESTSETS_DIR = Path(__file__).parent / "testsets"


async def fetch_chunks_as_langchain_docs(
    limit: int = 200,
    paper_title_filter: str | None = None,
) -> list[LangchainDocument]:
    """
    Fetch paper chunks from the database as LangChain Documents.

    Uses raw SQL to avoid loading 3072-dim embedding vectors into memory.

    Args:
        limit: Maximum number of chunks to fetch.
        paper_title_filter: Optional filter to only fetch chunks from a specific paper.

    Returns:
        List of LangchainDocument objects with page_content and metadata.
    """
    query = "SELECT text, metadata_, node_id FROM data_paper_chunks"
    params: dict = {}

    if paper_title_filter:
        query += " WHERE metadata_->>'title' ILIKE :title"
        params["title"] = f"%{paper_title_filter}%"

    query += " LIMIT :limit"
    params["limit"] = limit

    documents = []
    async with get_session_context() as session:
        result = await session.execute(text(query), params)
        rows = result.fetchall()

        for row in rows:
            chunk_text = row[0]
            metadata = row[1] if row[1] else {}
            node_id = row[2]

            if not chunk_text or not chunk_text.strip():
                continue

            # Build clean metadata for RAGAS
            doc_metadata = {
                "node_id": node_id,
                "title": metadata.get("title", "Unknown"),
            }
            if "doc_items" in metadata:
                doc_items = metadata["doc_items"]
                if doc_items and doc_items[0].get("prov"):
                    doc_metadata["page"] = doc_items[0]["prov"][0].get("page_no", "N/A")

            documents.append(
                LangchainDocument(
                    page_content=chunk_text.strip(),
                    metadata=doc_metadata,
                )
            )

    logger.info("Fetched %d chunks from database", len(documents))
    return documents


async def generate_testset(
    testset_size: int = 20,
    limit_chunks: int = 200,
    paper_title_filter: str | None = None,
) -> Path:
    """
    Generate a synthetic test set from database chunks using RAGAS.

    Args:
        testset_size: Number of test questions to generate.
        limit_chunks: Maximum chunks to fetch from the database.
        paper_title_filter: Optional filter for specific paper title.

    Returns:
        Path to the saved CSV test set file.
    """
    from ragas.testset import TestsetGenerator

    setup_langsmith_tracing("testset-generation")

    logger.info(
        "Generating testset: size=%d, chunk_limit=%d, filter=%s",
        testset_size,
        limit_chunks,
        paper_title_filter,
    )

    documents = await fetch_chunks_as_langchain_docs(limit_chunks, paper_title_filter)
    if not documents:
        raise ValueError("No chunks found in database. Run paper ingestion first.")

    generator = TestsetGenerator(
        llm=get_evaluator_llm(),
        embedding_model=get_evaluator_embeddings(),
    )

    # Use generate_with_chunks: our DB chunks are pre-chunked, so we skip
    # HeadlinesExtractor/HeadlineSplitter (which cause "headlines not found" errors
    # when documents are short or already chunked)
    testset = generator.generate_with_chunks(
        chunks=documents,
        testset_size=testset_size,
    )

    # Save as timestamped CSV
    TESTSETS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = TESTSETS_DIR / f"testset_{timestamp}.csv"

    df = testset.to_pandas()
    df.to_csv(output_path, index=False)

    logger.info("Testset saved to %s (%d rows)", output_path, len(df))
    print(f"Testset saved: {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate RAGAS synthetic test set")
    parser.add_argument("--size", type=int, default=20, help="Number of test questions")
    parser.add_argument("--limit", type=int, default=200, help="Max chunks to fetch")
    parser.add_argument("--paper", type=str, default=None, help="Filter by paper title")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(generate_testset(
        testset_size=args.size,
        limit_chunks=args.limit,
        paper_title_filter=args.paper,
    ))


if __name__ == "__main__":
    main()
