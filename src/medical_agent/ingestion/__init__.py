"""Medical paper ingestion pipeline."""

from .metadata import MedicalMetadata, extract_medical_metadata, stamp_metadata_on_nodes
from .pipeline import (
    MedicalIngestionPipeline,
    PipelineConfig,
    PipelineResult,
    process_pdf_llamaindex,
)

__all__ = [
    "MedicalIngestionPipeline",
    "PipelineConfig",
    "PipelineResult",
    "process_pdf_llamaindex",
    "MedicalMetadata",
    "extract_medical_metadata",
    "stamp_metadata_on_nodes",
]
