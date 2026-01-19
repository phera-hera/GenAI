"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Any


class BirthControlInfo(BaseModel):
    """Birth control information."""

    general: str | None = Field(
        None,
        description="No control, Stopped in last 3 months, or Emergency contraception in last 7 days",
    )
    pill: str | None = Field(None, description="Combined pill or Progestin-only pill")
    iud: str | None = Field(None, description="Hormonal IUD or Copper IUD")
    other_methods: list[str] = Field(
        default_factory=list,
        description="Implant, Injection, Vaginal ring, Patch",
    )
    permanent: list[str] = Field(
        default_factory=list, description="Tubal ligation"
    )


class FertilityJourneyInfo(BaseModel):
    """Fertility journey information."""

    current_status: str | None = Field(
        None,
        description="Pregnant, Had baby (last 12 months), Not able to get pregnant, or Trying to conceive",
    )
    fertility_treatments: list[str] = Field(
        default_factory=list,
        description="Ovulation induction, IUI, IVF, Egg freezing, Luteal progesterone",
    )


class SymptomsInfo(BaseModel):
    """Symptoms information."""

    discharge: list[str] = Field(
        default_factory=list,
        description="No discharge, Creamy, Sticky, Egg white, Clumpy white, Grey and watery, Yellow/Green, Red/Brown",
    )
    vulva_vagina: list[str] = Field(
        default_factory=list, description="Dry, Itchy"
    )
    smell: list[str] = Field(
        default_factory=list,
        description="Fishy, Sour, Chemical-like, Very strong or rotten",
    )
    urine: list[str] = Field(
        default_factory=list, description="Frequent urination, Burning sensation"
    )


class QueryRequest(BaseModel):
    """Request schema for pH analysis query from mobile app."""

    # Required
    ph_value: float = Field(
        ..., ge=0.0, le=14.0, description="pH value from test strip (0-14) - REQUIRED"
    )

    # Optional: Basic info
    age: int | None = Field(None, ge=0, le=120, description="User's age")

    # Optional: Medical background
    diagnoses: list[str] = Field(
        default_factory=list,
        description="Selected diagnoses related to hormones (multiselect)",
    )
    ethnic_backgrounds: list[str] = Field(
        default_factory=list,
        description="Selected ethnic background(s) (multiselect)",
    )

    # Optional: Hormone status
    menstrual_cycle: str | None = Field(
        None,
        description="Regular, Irregular, No period for 12+ months, Never had period, Perimenopause, or Postmenopause",
    )
    birth_control: BirthControlInfo | None = Field(
        None, description="Birth control methods"
    )
    hormone_therapy: list[str] = Field(
        default_factory=list,
        description="Estrogen only or Estrogen + Progestin (multiselect)",
    )
    hrt: list[str] = Field(
        default_factory=list,
        description="Hormone replacement therapy types - Testosterone, Estrogen blocker, Puberty blocker (multiselect)",
    )

    # Optional: Fertility
    fertility_journey: FertilityJourneyInfo | None = Field(
        None, description="Fertility journey information"
    )

    # Optional: Symptoms
    symptoms: SymptomsInfo | None = Field(None, description="Current symptoms")

    # Optional: Notes (NOT sent to agent, just for app context)
    notes: str | None = Field(
        None,
        description="User's free text notes - NOT included in medical analysis",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ph_value": 4.8,
                    "age": 28,
                    "diagnoses": ["PCOS"],
                    "ethnic_backgrounds": ["South Asian"],
                    "menstrual_cycle": "Irregular",
                    "birth_control": {"pill": "Combined pill"},
                    "symptoms": {
                        "discharge": ["Creamy"],
                        "vulva_vagina": ["Itchy"],
                        "smell": [],
                        "urine": [],
                    },
                    "notes": "Feeling stressed lately",
                }
            ]
        }
    }


class CitationResponse(BaseModel):
    """Citation from a research paper."""

    paper_id: str
    title: str | None
    authors: str | None
    doi: str | None = None
    relevant_section: str | None = None


class QueryResponse(BaseModel):
    """Response schema for pH analysis query."""

    session_id: str = Field(..., description="Unique session identifier")
    ph_value: float = Field(..., description="The analyzed pH value")
    risk_level: str = Field(
        ..., description="Risk assessment level (NORMAL, MONITOR, CONCERNING, URGENT)"
    )

    summary: str = Field(..., description="Brief summary of the analysis")
    main_content: str = Field(..., description="Detailed analysis content")
    personalized_insights: list[str] = Field(
        default_factory=list, description="Insights based on user profile"
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Recommended next steps"
    )
    disclaimers: str = Field(..., description="Medical disclaimers")

    citations: list[CitationResponse] = Field(
        default_factory=list, description="Research paper citations"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "abc123",
                    "ph_value": 4.8,
                    "risk_level": "MONITOR",
                    "summary": "Your pH is slightly elevated at 4.8",
                    "main_content": "A vaginal pH of 4.8 is slightly above...",
                    "personalized_insights": ["Based on your symptoms..."],
                    "next_steps": ["Continue monitoring", "Track symptoms"],
                    "disclaimers": "This is not medical advice...",
                    "citations": [],
                    "processing_time_ms": 1500,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")


class PaperInfo(BaseModel):
    """Information about an ingested paper."""

    id: str
    title: str
    authors: str | None
    doi: str | None
    chunk_count: int
    is_processed: bool


class PapersListResponse(BaseModel):
    """Response for listing papers."""

    papers: list[PaperInfo]
    total: int


# Ingestion Schemas


class IngestionRequest(BaseModel):
    """Request schema for ingesting papers."""

    gcp_paths: list[str] = Field(
        default_factory=list,
        description="List of GCP Cloud Storage paths (e.g., ['gs://bucket/paper1.pdf', 'paper2.pdf'])",
    )
    list_bucket: bool = Field(
        default=False,
        description="If True, list all PDFs in bucket and ingest all. Ignores gcp_paths.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, validate PDFs without storing embeddings.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gcp_paths": ["gs://femtech-medical-papers/paper1.pdf"],
                    "list_bucket": False,
                    "dry_run": False,
                },
                {
                    "gcp_paths": [],
                    "list_bucket": True,
                    "dry_run": False,
                },
            ]
        }
    }


class IngestionStageResult(BaseModel):
    """Result of a single ingestion stage."""

    stage: str = Field(..., description="Stage name (parse, chunk, embed, store, etc.)")
    success: bool = Field(..., description="Whether stage succeeded")
    duration_ms: int = Field(..., description="Duration in milliseconds")
    error: str | None = Field(None, description="Error message if failed")


class IngestionResultDetail(BaseModel):
    """Detailed result for a single paper ingestion."""

    paper_path: str = Field(..., description="GCP path of ingested paper")
    paper_id: str | None = Field(None, description="ID of ingested paper")
    success: bool = Field(..., description="Whether ingestion succeeded")
    total_duration_ms: int = Field(..., description="Total ingestion time")
    chunk_count: int = Field(0, description="Number of chunks created")
    metadata_found: dict[str, int] = Field(
        default_factory=dict,
        description="Count of metadata items found (e.g., {'diagnoses': 2, 'symptoms': 3})",
    )
    stages: list[IngestionStageResult] = Field(
        default_factory=list, description="Results of each pipeline stage"
    )
    error: str | None = Field(None, description="Error message if failed")


class IngestionResponse(BaseModel):
    """Response schema for ingestion request."""

    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(
        ...,
        description="Overall status (COMPLETED, PARTIAL_SUCCESS, FAILED)",
    )
    total_papers_processed: int = Field(..., description="Total papers processed")
    successful: int = Field(..., description="Successful ingestions")
    failed: int = Field(..., description="Failed ingestions")
    total_chunks_created: int = Field(..., description="Total chunks across all papers")
    total_duration_ms: int = Field(..., description="Total processing time")
    details: list[IngestionResultDetail] = Field(
        default_factory=list, description="Per-paper ingestion details"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "ingest-20240101-123456",
                    "status": "COMPLETED",
                    "total_papers_processed": 2,
                    "successful": 2,
                    "failed": 0,
                    "total_chunks_created": 245,
                    "total_duration_ms": 45000,
                    "details": [
                        {
                            "paper_path": "gs://bucket/paper1.pdf",
                            "paper_id": "paper-001",
                            "success": True,
                            "total_duration_ms": 22000,
                            "chunk_count": 120,
                            "metadata_found": {
                                "diagnoses": 2,
                                "symptoms": 3,
                                "ethnicities": 1,
                            },
                            "stages": [
                                {
                                    "stage": "parse",
                                    "success": True,
                                    "duration_ms": 5000,
                                    "error": None,
                                },
                            ],
                            "error": None,
                        }
                    ],
                }
            ]
        }
    }
