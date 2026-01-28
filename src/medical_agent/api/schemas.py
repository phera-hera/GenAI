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
        description='Strong and unpleasant ("fishy"), Sour, Chemical-like, Very strong or rotten',
    )
    urine: list[str] = Field(
        default_factory=list, description="Frequent urination, Burning sensation"
    )
    notes: str | None = Field(
        None,
        description="User's free text notes about additional symptoms or how they've been feeling",
    )


class QueryRequest(BaseModel):
    """Request schema for pH analysis query from mobile app."""

    # Required
    ph_value: float = Field(
        ..., ge=0.0, le=14.0, description="pH value from test strip (0-14) - REQUIRED"
    )

    # Optional: Conversation context
    user_message: str | None = Field(
        None, description="User's actual question (for follow-ups). If None, auto-generates initial query."
    )
    session_id: str | None = Field(
        None, description="Session ID for conversation continuity. If None, creates new session."
    )

    # Optional: Basic info
    age: int | None = Field(None, ge=0, le=120, description="User's age")

    # Optional: Medical background
    diagnoses: list[str] = Field(
        default_factory=list,
        description="Diagnoses related to hormones: Adenomyosis, Amenorrhea, Cushing's syndrome, Diabetes, Endometriosis, Intersex status, Thyroid disorder, Uterine fibroids, Polycystic ovary syndrome (PCOS), Premature ovarian insufficiency (POI)",
    )
    ethnic_backgrounds: list[str] = Field(
        default_factory=list,
        description="Ethnic backgrounds: African / Black, North African, Arab, Middle Eastern, East Asian, South Asian, Southeast Asian, Central Asian / Caucasus, Latin American / Latina / Latinx / Hispanic, Sinti / Roma, White / Caucasian / European, Mixed / Multiple ancestries, Other",
    )

    # Optional: Hormone status
    menstrual_cycle: str | None = Field(
        None,
        description="Regular, Irregular, No period for 12+ months, Never had a period, Perimenopausal, Postmenopausal",
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ph_value": 4.8,
                    "age": 28,
                    "diagnoses": ["Polycystic ovary syndrome (PCOS)", "Endometriosis"],
                    "ethnic_backgrounds": ["South Asian", "East Asian"],
                    "menstrual_cycle": "Irregular",
                    "birth_control": {"pill": "Combined pill"},
                    "symptoms": {
                        "discharge": ["Creamy"],
                        "vulva_vagina": ["Itchy"],
                        "smell": [],
                        "urine": [],
                        "notes": "Feeling stressed lately",
                    },
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
    agent_reply: str = Field(..., description="Complete medical analysis response")
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
                    "agent_reply": "Your pH reading of 4.8 is slightly elevated. A vaginal pH of 4.8 is slightly above the normal range...",
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


