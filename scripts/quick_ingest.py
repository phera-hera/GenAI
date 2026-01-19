#!/usr/bin/env python3
"""
Quick ingestion script for medical papers.

Usage:
    # Ingest specific papers
    python scripts/quick_ingest.py --papers paper1.pdf paper2.pdf

    # List and ingest ALL papers in bucket
    python scripts/quick_ingest.py --all

    # Dry run (validate without storing)
    python scripts/quick_ingest.py --papers paper1.pdf --dry-run
"""

import asyncio
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def ingest_papers(
    papers: list[str] | None = None,
    list_bucket: bool = False,
    dry_run: bool = False,
    api_url: str = "http://localhost:8000/api/v1/ingest",
) -> dict:
    """
    Ingest papers via the REST API.

    Args:
        papers: List of paper paths to ingest
        list_bucket: If True, ingest all papers in bucket
        dry_run: If True, validate without storing
        api_url: Backend API URL

    Returns:
        Ingestion result
    """
    if not papers and not list_bucket:
        raise ValueError("Must specify either papers or list_bucket=True")

    logger.info(
        f"Triggering ingestion: "
        f"papers={len(papers) if papers else 'all'}, "
        f"dry_run={dry_run}"
    )

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response = await client.post(
                api_url,
                json={
                    "gcp_paths": papers or [],
                    "list_bucket": list_bucket,
                    "dry_run": dry_run,
                },
            )
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to API: {e}")
            logger.error(f"Make sure backend is running at {api_url}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"API error: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                logger.error(f"Details: {error_detail}")
            except Exception:
                logger.error(f"Response: {e.response.text}")
            raise


def print_results(result: dict) -> int:
    """
    Print ingestion results in a nice format.

    Returns:
        Exit code (0 if all succeeded, 1 if any failed)
    """
    print("\n" + "=" * 80)
    print("INGESTION RESULTS")
    print("=" * 80)

    print(f"\nRequest ID: {result['request_id']}")
    print(f"Status: {result['status']}")
    print(f"Duration: {result['total_duration_ms'] / 1000:.1f}s")
    print()

    print(f"Papers Processed: {result['total_papers_processed']}")
    print(f"  ✓ Successful: {result['successful']}")
    print(f"  ✗ Failed: {result['failed']}")
    print()

    print(f"Total Chunks Created: {result['total_chunks_created']}")

    print("\n" + "-" * 80)
    print("PER-PAPER DETAILS")
    print("-" * 80)

    for detail in result["details"]:
        status_icon = "✓" if detail["success"] else "✗"
        print(f"\n{status_icon} {detail['paper_path']}")

        if detail["success"]:
            print(f"   Paper ID: {detail['paper_id']}")
            print(f"   Duration: {detail['total_duration_ms'] / 1000:.1f}s")
            print(f"   Chunks: {detail['chunk_count']}")

            if detail["metadata_found"]:
                print(f"   Metadata Extracted:")
                for key, count in detail["metadata_found"].items():
                    print(f"     • {key}: {count}")

            if detail["stages"]:
                print(f"   Pipeline Stages:")
                for stage in detail["stages"]:
                    stage_status = "✓" if stage["success"] else "✗"
                    print(
                        f"     {stage_status} {stage['stage']}: "
                        f"{stage['duration_ms']}ms"
                    )
        else:
            print(f"   Error: {detail['error']}")

            if detail["stages"]:
                print(f"   Failed at:")
                for stage in detail["stages"]:
                    if not stage["success"]:
                        print(f"     ✗ {stage['stage']}: {stage['error']}")

    print("\n" + "=" * 80)

    # Return exit code
    if result["failed"] > 0:
        print(f"\n⚠️  {result['failed']} paper(s) failed")
        return 1
    else:
        print("\n✓ All papers ingested successfully!")
        return 0


async def main():
    """Main entry point."""
    parser = ArgumentParser(description="Ingest medical papers into vector store")

    parser.add_argument(
        "--papers",
        nargs="+",
        help="Papers to ingest (e.g., --papers paper1.pdf paper2.pdf)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all PDFs in configured GCP bucket",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate papers without storing embeddings",
    )

    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1/ingest",
        help="Backend API URL",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.papers and not args.all:
        parser.error("Must specify either --papers or --all")

    try:
        # Call API
        result = await ingest_papers(
            papers=args.papers,
            list_bucket=args.all,
            dry_run=args.dry_run,
            api_url=args.api_url,
        )

        # Print results and exit
        exit_code = print_results(result)
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n✗ Ingestion cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
