"""Medical metadata types extracted from research papers."""

from dataclasses import dataclass, field


@dataclass
class ExtractedMetadata:
    """Medical metadata extracted from a research paper."""

    ethnicities: list[str] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    menstrual_status: list[str] = field(default_factory=list)
    birth_control: list[str] = field(default_factory=list)
    hormone_therapy: list[str] = field(default_factory=list)
    fertility_treatments: list[str] = field(default_factory=list)
    age_mentioned: bool = False
    age_range: str | None = None
    confidence: float = 0.0
