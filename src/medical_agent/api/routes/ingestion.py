"""
Ingestion API Routes

Provides endpoints for ingesting and managing medical papers in the vector store.
Supports GCP Cloud Storage integration for automated paper ingestion.

Uses LlamaIndex-based ingestion pipeline with Docling for PDF parsing.
"""

import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from medical_agent.api.schemas import (
    ErrorResponse,
    IngestionRequest,
    IngestionResponse,
    IngestionResultDetail,
    IngestionStageResult,
)
from medical_agent.core.config import settings
from medical_agent.infrastructure.gcp_storage import get_storage_client

from medical_agent.ingestion.pipeline import MedicalIngestionPipeline, process_pdf_llamaindex

logger = logging.getLogger(__name__)
logger.info("Using LlamaIndex ingestion pipeline")

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        500: {"model": ErrorResponse, "description": "Ingestion failed"},
    },
    summary="Ingest medical papers",
    description="""
    Trigger ingestion of medical papers from GCP Cloud Storage.

    **Options**:
    - `gcp_paths`: List specific paper paths from GCP (e.g., `gs://bucket/paper.pdf` or just `paper.pdf`)
    - `list_bucket`: If true, automatically list and ingest ALL PDFs in the configured bucket
    - `dry_run`: Validate papers without storing embeddings (for testing)

    **Process**:
    1. Download PDF from GCP
    2. Parse with Docling (vision-based, handles scans + tables)
    3. Extract metadata (ethnicity, diagnoses, symptoms, etc.)
    4. Chunk hierarchically (respects section boundaries)
    5. Embed with Azure OpenAI (text-embedding-3-large)
    6. Store in PostgreSQL with pgvector

    **Response** includes:
    - Per-paper ingestion details
    - Metadata extracted (diagnosis, symptom counts, etc.)
    - Chunk counts and processing time
    - Error details for any failures

    **Ideal Use**:
    - Manual: Call this endpoint after uploading papers to GCP
    - Automated: Cloud Functions triggered by Cloud Storage events (see docs)
    """,
)
async def ingest_papers(request: IngestionRequest) -> IngestionResponse:
    """
    Ingest medical papers from GCP Cloud Storage.

    Args:
        request: Ingestion request with GCP paths or bucket listing flag

    Returns:
        Detailed ingestion results for each paper
    """
    # Verify GCP is configured
    if not settings.is_gcp_storage_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "SERVICE_NOT_CONFIGURED",
                "message": "GCP Cloud Storage is not configured. Paper ingestion unavailable.",
            },
        )

    # Verify database is configured
    try:
        from sqlalchemy import text

        from medical_agent.infrastructure.database.session import get_session_context

        async with get_session_context() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        logger.exception("Database connection check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "DATABASE_UNAVAILABLE",
                "message": f"Database is not accessible: {str(e)}",
            },
        )

    request_id = f"ingest-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    logger.info(f"Starting ingestion request: {request_id}")

    # Determine papers to ingest
    papers_to_ingest: list[str] = []

    try:
        storage_client = get_storage_client()

        if request.list_bucket:
            # List all PDFs in bucket
            logger.info(f"Listing all PDFs in bucket: {settings.gcp_bucket_name}")
            papers_to_ingest = storage_client.list_pdfs()
            logger.info(f"Found {len(papers_to_ingest)} papers in bucket")

        elif request.gcp_paths:
            # Normalize paths
            for path in request.gcp_paths:
                # Handle "gs://bucket/..." or just "filename.pdf"
                if path.startswith("gs://"):
                    # Extract just the filename
                    papers_to_ingest.append(path.split("/")[-1])
                else:
                    papers_to_ingest.append(path)

            logger.info(f"Will ingest {len(papers_to_ingest)} specified papers")

        else:
            raise ValueError("Must specify either gcp_paths or set list_bucket=true")

    except Exception as e:
        logger.exception(f"Failed to determine papers to ingest: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_REQUEST",
                "message": f"Failed to determine papers to ingest: {str(e)}",
            },
        )

    if not papers_to_ingest:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "NO_PAPERS_FOUND",
                "message": "No papers found to ingest",
            },
        )

    logger.info(f"Request {request_id}: Processing {len(papers_to_ingest)} papers")

    # Process each paper
    start_time = time.time()
    results: list[IngestionResultDetail] = []
    successful = 0
    failed = 0
    total_chunks = 0

    for paper_path in papers_to_ingest:
        paper_result = await _ingest_single_paper(
            paper_path=paper_path,
            request_id=request_id,
            dry_run=request.dry_run,
        )

        results.append(paper_result)

        if paper_result.success:
            successful += 1
            total_chunks += paper_result.chunk_count
        else:
            failed += 1

    total_duration = int((time.time() - start_time) * 1000)

    # Determine overall status
    if failed == 0:
        overall_status = "COMPLETED"
    elif successful > 0:
        overall_status = "PARTIAL_SUCCESS"
    else:
        overall_status = "FAILED"

    logger.info(
        f"Request {request_id} completed: {successful} succeeded, {failed} failed "
        f"({total_duration}ms, {total_chunks} chunks)"
    )

    return IngestionResponse(
        request_id=request_id,
        status=overall_status,
        total_papers_processed=len(papers_to_ingest),
        successful=successful,
        failed=failed,
        total_chunks_created=total_chunks,
        total_duration_ms=total_duration,
        details=results,
    )


async def _ingest_single_paper(
    paper_path: str, request_id: str, dry_run: bool = False
) -> IngestionResultDetail:
    """
    Ingest a single paper from GCP.

    Args:
        paper_path: GCP path (can be gs://... or just filename)
        request_id: Ingestion request ID for logging
        dry_run: If True, validate without storing

    Returns:
        Detailed result for this paper
    """
    start_time = time.time()
    paper_id = None
    stages: list[IngestionStageResult] = []
    metadata_counts: dict[str, int] = {}

    try:
        logger.info(f"[{request_id}] Ingesting: {paper_path}")

        # Download PDF from GCP
        stage_start = time.time()
        storage_client = get_storage_client()

        # Normalize path
        gcp_path = paper_path
        if not gcp_path.startswith("gs://"):
            gcp_path = f"gs://{settings.gcp_bucket_name}/{paper_path}"

        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: storage_client.download_pdf_to_path(paper_path, tmp_path),
            )

            stages.append(
                IngestionStageResult(
                    stage="download",
                    success=True,
                    duration_ms=int((time.time() - stage_start) * 1000),
                )
            )

            logger.info(f"[{request_id}] Downloaded: {paper_path}")

        except Exception as e:
            logger.exception(f"[{request_id}] Download failed for {paper_path}: {e}")
            stages.append(
                IngestionStageResult(
                    stage="download",
                    success=False,
                    duration_ms=int((time.time() - stage_start) * 1000),
                    error=str(e),
                )
            )
            raise

        # Run ingestion pipeline
        try:
            stage_start = time.time()

            # Read file content
            with open(tmp_path, "rb") as f:
                pdf_content = f.read()

            # Use LlamaIndex pipeline
            result = await process_pdf_llamaindex(
                pdf_content=pdf_content,
                gcp_path=gcp_path,
            )

            # Extract pipeline results
            pipeline_duration = int((time.time() - stage_start) * 1000)

            # Parse pipeline result stages
            if hasattr(result, "stages_completed"):
                for stage_name, stage_result in result.stages_completed.items():
                    if isinstance(stage_result, dict):
                        stages.append(
                            IngestionStageResult(
                                stage=stage_name,
                                success=stage_result.get("success", True),
                                duration_ms=stage_result.get("duration_ms", 0),
                                error=stage_result.get("error"),
                            )
                        )

            # Extract metadata counts
            if hasattr(result, "metadata") and result.metadata:
                metadata = result.metadata
                if hasattr(metadata, "ethnicities"):
                    metadata_counts["ethnicities"] = len(metadata.ethnicities)
                if hasattr(metadata, "diagnoses"):
                    metadata_counts["diagnoses"] = len(metadata.diagnoses)
                if hasattr(metadata, "symptoms"):
                    metadata_counts["symptoms"] = len(metadata.symptoms)
                if hasattr(metadata, "hormone_therapy"):
                    metadata_counts["hormone_therapy"] = len(metadata.hormone_therapy)
                if hasattr(metadata, "birth_control"):
                    metadata_counts["birth_control"] = len(metadata.birth_control)

            stages.append(
                IngestionStageResult(
                    stage="pipeline",
                    success=True,
                    duration_ms=pipeline_duration,
                )
            )

            paper_id = str(result.paper_id) if hasattr(result, "paper_id") and result.paper_id else None
            chunk_count = result.chunk_count if hasattr(result, "chunk_count") else 0

            logger.info(
                f"[{request_id}] Ingested successfully: {paper_path} "
                f"({chunk_count} chunks, metadata: {metadata_counts})"
            )

        except Exception as e:
            logger.exception(f"[{request_id}] Pipeline failed for {paper_path}: {e}")
            stages.append(
                IngestionStageResult(
                    stage="pipeline",
                    success=False,
                    duration_ms=int((time.time() - stage_start) * 1000),
                    error=str(e),
                )
            )
            raise

        finally:
            # Cleanup temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

        # Success
        total_duration = int((time.time() - start_time) * 1000)

        return IngestionResultDetail(
            paper_path=paper_path,
            paper_id=paper_id,
            success=True,
            total_duration_ms=total_duration,
            chunk_count=chunk_count,
            metadata_found=metadata_counts,
            stages=stages,
            error=None,
        )

    except Exception as e:
        total_duration = int((time.time() - start_time) * 1000)
        logger.error(f"[{request_id}] Ingestion failed for {paper_path}: {e}")

        return IngestionResultDetail(
            paper_path=paper_path,
            paper_id=None,
            success=False,
            total_duration_ms=total_duration,
            chunk_count=0,
            metadata_found={},
            stages=stages,
            error=str(e),
        )
