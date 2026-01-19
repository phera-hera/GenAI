"""
Data Types for Metadata Extraction

Defines structured types for extracted medical metadata.
"""

from dataclasses import dataclass, field


@dataclass
class TableMetadata:
    """
    Metadata extracted for a single table.

    Contains a one-sentence summary describing the table's variables and purpose.
    """

    table_id: int
    summary: str
    confidence: float = 0.0


@dataclass
class ExtractedMetadata:
    """
    Medical metadata extracted from a research paper.

    All categories are present (may be empty lists). Only explicitly mentioned
    terms are included - no hallucination of relevance.

    Attributes:
        ethnicities: List of ethnicity terms found (e.g., ["African / Black"])
        diagnoses: List of hormone-related diagnoses (e.g., ["PCOS", "Endometriosis"])
        symptoms: List of symptoms mentioned (e.g., ["Vaginal Odor", "Itchy"])
        menstrual_status: List of menstrual cycle states
        birth_control: List of birth control types mentioned
        hormone_therapy: List of hormone therapies mentioned
        fertility_treatments: List of fertility treatments mentioned
        age_mentioned: Whether age is explicitly mentioned
        age_range: Age range if mentioned (e.g., "25-35")
        table_summaries: Dict mapping table_id to one-sentence summary
        confidence: Overall confidence score from LLM (0.0-1.0)
    """

    # Medical context categories
    ethnicities: list[str] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    menstrual_status: list[str] = field(default_factory=list)
    birth_control: list[str] = field(default_factory=list)
    hormone_therapy: list[str] = field(default_factory=list)
    fertility_treatments: list[str] = field(default_factory=list)

    # Age information
    age_mentioned: bool = False
    age_range: str | None = None

    # Table summaries
    table_summaries: dict[int, str] = field(default_factory=dict)

    # Extraction quality
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage in chunk_metadata."""
        return {
            "ethnicities": self.ethnicities,
            "diagnoses": self.diagnoses,
            "symptoms": self.symptoms,
            "menstrual_status": self.menstrual_status,
            "birth_control": self.birth_control,
            "hormone_therapy": self.hormone_therapy,
            "fertility_treatments": self.fertility_treatments,
            "age_mentioned": self.age_mentioned,
            "age_range": self.age_range,
            "confidence": self.confidence,
        }

    def is_empty(self) -> bool:
        """Check if no metadata was extracted."""
        return (
            not self.ethnicities
            and not self.diagnoses
            and not self.symptoms
            and not self.menstrual_status
            and not self.birth_control
            and not self.hormone_therapy
            and not self.fertility_treatments
            and not self.age_mentioned
            and not self.table_summaries
        )
