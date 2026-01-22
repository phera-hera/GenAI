"""
Batch Ingestion Script for Medical Research Papers

Downloads PDFs from GCP Cloud Storage and processes them through
the ingestion pipeline (parse → chunk → embed → store in pgvector).

Usage:
    python scripts/ingest_papers.py                    # Ingest all papers
    python scripts/ingest_papers.py --prefix papers/   # Filter by prefix
    python scripts/ingest_papers.py --limit 5          # Limit to 5 papers
    python scripts/ingest_papers.py --dry-run          # Preview without processing
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_agent.core.config import settings
from medical_agent.infrastructure.gcp_storage import get_storage_client
from medical_agent.infrastructure.database.session import init_db, get_session_context
from medical_agent.ingestion.pipeline import (
    MedicalIngestionPipeline,
    PipelineConfig,
    PipelineResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IngestionProgress:
    """Track progress of batch ingestion."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self.results: list[PipelineResult] = []

    def update(self, result: PipelineResult) -> None:
        self.completed += 1
        self.results.append(result)
        if result.success:
            self.succeeded += 1
        else:
            self.failed += 1

    def print_progress(self, current_file: str) -> None:
        pct = (self.completed / self.total) * 100 if self.total > 0 else 0
        logger.info(
            f"[{self.completed}/{self.total}] ({pct:.0f}%) "
            f"Success: {self.succeeded}, Failed: {self.failed} | "
            f"Current: {current_file}"
        )

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Total papers processed: {self.completed}")
        print(f"Succeeded: {self.succeeded}")
        print(f"Failed: {self.failed}")
        print("-" * 60)

        if self.results:
            total_chunks = sum(r.chunk_count for r in self.results)
            total_embedded = sum(r.embedded_count for r in self.results)
            total_stored = sum(r.stored_count for r in self.results)
            total_time = sum(r.total_time_ms for r in self.results)

            print(f"Total chunks created: {total_chunks}")
            print(f"Total embeddings generated: {total_embedded}")
            print(f"Total chunks stored: {total_stored}")
            print(f"Total processing time: {total_time / 1000:.1f}s")

        if self.failed > 0:
            print("\nFailed papers:")
            for r in self.results:
                if not r.success:
                    print(f"  - {r.gcp_path}: {', '.join(r.errors)}")

        print("=" * 60)


async def ingest_papers(
    prefix: str = "papers/",
    limit: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> IngestionProgress:
    """
    Ingest papers from GCP bucket.

    Args:
        prefix: GCP path prefix to filter papers
        limit: Maximum number of papers to process
        dry_run: If True, only list papers without processing
        skip_existing: Skip papers already in the database

    Returns:
        IngestionProgress with results
    """
    # Verify configuration
    if not settings.is_gcp_configured():
        raise ValueError(
            "GCP is not configured. Set GCP_PROJECT_ID and GCP_BUCKET_NAME in .env"
        )

    if not settings.is_azure_openai_configured():
        raise ValueError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and related vars in .env"
        )

    if not settings.is_llama_parser_configured():
        raise ValueError(
            "LlamaParser LVM mode is not configured. "
            "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT_NAME in .env"
        )

    # Initialize database
    logger.info("Initializing database...")
    await init_db()

    # Get storage client
    storage = get_storage_client()

    # Verify GCP connection
    logger.info(f"Connecting to GCP bucket: {settings.gcp_bucket_name}")
    storage.verify_connection()

    # List PDFs
    logger.info(f"Listing PDFs with prefix: {prefix}")
    pdf_files = storage.list_pdfs(prefix=prefix)

    if limit:
        pdf_files = pdf_files[:limit]

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    if not pdf_files:
        logger.warning("No PDF files found!")
        return IngestionProgress(0)

    # Preview mode
    if dry_run:
        print("\nDRY RUN - Papers to be processed:")
        print("-" * 60)
        for i, pdf in enumerate(pdf_files, 1):
            size_kb = pdf["size"] / 1024 if pdf["size"] else 0
            print(f"{i}. {pdf['name']} ({size_kb:.1f} KB)")
        print("-" * 60)
        print(f"Total: {len(pdf_files)} papers")
        return IngestionProgress(len(pdf_files))

    # Initialize pipeline
    pipeline = MedicalIngestionPipeline(
        config=PipelineConfig(
            max_chunk_tokens=512,
            extract_metadata=True,
            extract_table_summaries=True,
        )
    )

    progress = IngestionProgress(len(pdf_files))

    # Process each paper
    for pdf_info in pdf_files:
        gcp_path = pdf_info["name"]
        progress.print_progress(gcp_path)

        try:
            # Download PDF
            logger.info(f"Downloading: {gcp_path}")
            pdf_content = storage.download_pdf(gcp_path)

            # Process through pipeline
            async with get_session_context() as session:
                result = await pipeline.process_paper(
                    session=session,
                    pdf_content=pdf_content,
                    gcp_path=f"gs://{settings.gcp_bucket_name}/{gcp_path}",
                )

                # Commit the transaction
                await session.commit()

            progress.update(result)

            if result.success:
                logger.info(
                    f"SUCCESS: {result.paper_title or gcp_path} - "
                    f"{result.stored_count} chunks stored in {result.total_time_ms}ms"
                )
            else:
                logger.error(f"FAILED: {gcp_path} - {result.errors}")

        except Exception as e:
            logger.error(f"ERROR processing {gcp_path}: {e}")
            # Create a failed result
            from uuid import uuid4
            failed_result = PipelineResult(
                paper_id=uuid4(),
                paper_title=None,
                gcp_path=gcp_path,
                errors=[str(e)],
            )
            progress.update(failed_result)

    progress.print_summary()
    return progress


def main():
    parser = argparse.ArgumentParser(
        description="Ingest medical research papers from GCP bucket"
    )
    parser.add_argument(
        "--prefix",
        default="papers/",
        help="GCP path prefix to filter papers (default: papers/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview papers without processing",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process papers even if they exist in database",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            ingest_papers(
                prefix=args.prefix,
                limit=args.limit,
                dry_run=args.dry_run,
                skip_existing=not args.no_skip_existing,
            )
        )
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
