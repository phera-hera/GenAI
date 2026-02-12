"""
Chunk-level medical metadata extraction (production approach).

Extracts metadata from chunks after chunking. Uses paper_id as cache key
to ensure each paper gets unique metadata extraction.

Industry standard: Extract metadata during/after chunking phase, not before.
"""

import logging
from typing import Any, Sequence

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.extractors import BaseExtractor
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.llms.azure_openai import AzureOpenAI

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


class MedicalMetadata(BaseModel):
    """Medical metadata extracted from research papers."""

    ethnicities: list[str] = Field(
        default_factory=list,
        description=(
            "Ethnicity terms EXPLICITLY mentioned. Must be one of: "
            "African / Black, Asian, Caucasian, Hispanic / Latina, Middle Eastern, "
            "Mixed, Native American / Indigenous, North African, Pacific Islander, "
            "South Asian, Southeast Asian. Only extract if explicitly stated."
        )
    )

    diagnoses: list[str] = Field(
        default_factory=list,
        description=(
            "Hormone-related diagnoses EXPLICITLY mentioned. Must be one of: "
            "Adenomyosis, Endometriosis, Bacterial vaginosis, Yeast infection, "
            "Sexually transmitted infection, Polycystic ovary syndrome (PCOS), "
            "Premature ovarian insufficiency, Thyroid disorder, Fibroids (uterine myomas), "
            "Ovarian cysts, Pelvic inflammatory disease. Only extract if explicitly stated."
        )
    )

    symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "Symptoms EXPLICITLY mentioned. Must be one of: "
            "Creamy, Clear, Yellow, Green, Gray, Pink, Brown (discharge colors), "
            "Vaginal Odor, Itchy, Burning, Pelvic Pain, Swelling, Redness, "
            "Vaginal Dryness, Frequent Urination, Painful Urination. "
            "Only extract if explicitly stated."
        )
    )

    menstrual_status: list[str] = Field(
        default_factory=list,
        description=(
            "Menstrual cycle states EXPLICITLY mentioned. Must be one of: "
            "Premenstrual, Menstrual, Postmenstrual, Ovulation, Luteal Phase, "
            "Follicular Phase. Only extract if explicitly stated."
        )
    )

    birth_control: list[str] = Field(
        default_factory=list,
        description=(
            "Birth control types EXPLICITLY mentioned. Must be one of: "
            "Pill, IUD, Implant, Patch, Ring, Injection, Condom, Diaphragm, "
            "Sterilization. Only extract if explicitly stated."
        )
    )

    hormone_therapy: list[str] = Field(
        default_factory=list,
        description=(
            "Hormone therapies EXPLICITLY mentioned. Must be one of: "
            "HRT, Testosterone, Progesterone, Estrogen, Thyroid Medication. "
            "Only extract if explicitly stated."
        )
    )

    fertility_treatments: list[str] = Field(
        default_factory=list,
        description=(
            "Fertility treatments EXPLICITLY mentioned. Must be one of: "
            "IVF, IUI, Clomiphene, Letrozole, Gonadotropins, Ovulation Induction. "
            "Only extract if explicitly stated."
        )
    )

    age_mentioned: bool = Field(
        default=False,
        description="Whether age or age range is explicitly mentioned in the text"
    )

    age_range: str | None = Field(
        default=None,
        description="Age range if mentioned (e.g., '25-35', '18-45'). None if not mentioned."
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for extraction quality (0.0-1.0). Higher is better."
    )


EXTRACTION_PROMPT = """You are extracting metadata from research papers for a women's health application.

CRITICAL RULES:
1. ONLY extract information that is EXPLICITLY MENTIONED in the text
2. DO NOT infer, extrapolate, or hallucinate
3. Use EXACT standardized terms from field descriptions

MEDICAL METADATA (standardized terms only):
- ethnicities: African/Black, Asian, Caucasian, Hispanic/Latina, Middle Eastern, Mixed, Native American/Indigenous, North African, Pacific Islander, South Asian, Southeast Asian
- diagnoses: Adenomyosis, Endometriosis, Bacterial vaginosis, Yeast infection, STI, PCOS, Premature ovarian insufficiency, Thyroid disorder, Fibroids, Ovarian cysts, Pelvic inflammatory disease
- symptoms: Discharge colors (Creamy, Clear, Yellow, Green, Gray, Pink, Brown), Vaginal Odor, Itchy, Burning, Pelvic Pain, Swelling, Redness, Vaginal Dryness, Frequent Urination, Painful Urination
- birth_control: Pill, IUD, Implant, Patch, Ring, Injection, Condom, Diaphragm, Sterilization
- hormone_therapy: HRT, Testosterone, Progesterone, Estrogen, Thyroid Medication
- fertility_treatments: IVF, IUI, Clomiphene, Letrozole, Gonadotropins, Ovulation Induction
- menstrual_status: Premenstrual, Menstrual, Postmenstrual, Ovulation, Luteal Phase, Follicular Phase
- age_range: Age/range if mentioned (e.g., "25-35", "18-45")

MAPPING:
- "PCOS" → "Polycystic ovary syndrome (PCOS)"
- "candida" → "Yeast infection"
- "birth control pill" → "Pill"

OUTPUT:
- Return empty arrays for categories with no mentions
- Confidence: 0.0-1.0 based on clarity (higher = clearer)
- Focus on abstract, introduction, methods, and results sections

Extract metadata from the following text:"""


class MedicalMetadataExtractor(BaseExtractor):
    """
    Chunk-level medical metadata extractor (production pattern).

    Extracts metadata from chunks after chunking, using paper_id as cache key.
    Each paper gets one LLM call, metadata is stamped on all chunks.
    """

    def __init__(self, llm: LLM | None = None, **kwargs):
        """Initialize the extractor with an LLM."""
        super().__init__(**kwargs)

        if llm is None:
            model = getattr(
                settings,
                "metadata_extraction_model",
                settings.azure_openai_deployment_name,
            )
            llm = AzureOpenAI(
                model=model,
                deployment_name=model,
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                temperature=0.0,
            )
            logger.info(f"Created Azure OpenAI LLM for metadata extraction: {model}")

        # Store using object.__setattr__ to avoid Pydantic validation
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_cache", {})

        prompt_template = ChatPromptTemplate.from_messages([
            ("user", EXTRACTION_PROMPT + "\n\n{text}")
        ])
        object.__setattr__(self, "_prompt_template", prompt_template)

        logger.info("Created medical metadata extractor (chunk-level)")

    def _extract_relevant_text(self, nodes: Sequence[BaseNode]) -> str:
        """
        Extract relevant text from abstract/methods chunks.

        Uses Docling's headings metadata to find relevant sections.
        """
        parts = []

        for node in nodes[:10]:  # Check first 10 chunks
            content = node.get_content()
            headings = node.metadata.get("headings") or []

            # Convert headings to searchable string
            heading_text = " > ".join(headings).lower() if headings else ""

            # Prioritize abstract and methods sections
            if "abstract" in heading_text:
                parts.append(f"ABSTRACT:\n{content[:2000]}")
            elif "method" in heading_text or "material" in heading_text:
                parts.append(f"METHODS:\n{content[:2000]}")
            elif "introduction" in heading_text and len(parts) < 2:
                parts.append(f"INTRODUCTION:\n{content[:1500]}")
            elif "result" in heading_text and len(parts) < 3:
                parts.append(f"RESULTS:\n{content[:1500]}")

            # Stop if we have enough
            if len(parts) >= 3 or sum(len(p) for p in parts) >= 6000:
                break

        # Fallback: if no sections found, take first few chunks
        if not parts and nodes:
            for node in nodes[:3]:
                parts.append(node.get_content()[:2000])

        return "\n\n".join(parts)

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        """Extract medical metadata and stamp on all nodes."""
        if not nodes:
            return []

        try:
            # Use paper_id as cache key (CRITICAL FIX)
            paper_id = nodes[0].metadata.get("paper_id")

            if not paper_id:
                logger.error("No paper_id found in chunk metadata — cannot cache properly")
                cache_key = "unknown"
            else:
                cache_key = paper_id

            # Check cache
            if cache_key in self._cache:
                logger.debug(f"Using cached metadata for paper {cache_key}")
                cached_result = self._cache[cache_key]
            else:
                logger.info(f"Extracting metadata for paper {cache_key} ({len(nodes)} chunks)")

                if not getattr(settings, "metadata_extraction_enabled", True):
                    logger.warning("Metadata extraction is disabled in settings")
                    cached_result = self._empty_metadata()
                else:
                    # Extract relevant text from chunks
                    text = self._extract_relevant_text(nodes)

                    if not text.strip():
                        logger.warning(f"No text available for extraction (paper {cache_key})")
                        cached_result = self._empty_metadata()
                    else:
                        # Call LLM with structured output
                        result = await self._llm.astructured_predict(
                            output_cls=MedicalMetadata,
                            prompt=self._prompt_template,
                            text=text,
                        )

                        cached_result = result.model_dump()

                        total_terms = sum(
                            len(v) for k, v in cached_result.items()
                            if isinstance(v, list)
                        )
                        logger.info(
                            f"Extracted {total_terms} terms for paper {cache_key}, "
                            f"confidence: {cached_result.get('confidence', 0.0):.2f}"
                        )

                # Cache result by paper_id
                self._cache[cache_key] = cached_result

            # Return same metadata for all chunks
            return [cached_result.copy() for _ in nodes]

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}", exc_info=True)
            return [self._empty_metadata() for _ in nodes]

    def _empty_metadata(self) -> dict[str, Any]:
        """Return empty metadata structure."""
        return {
            "ethnicities": [],
            "diagnoses": [],
            "symptoms": [],
            "menstrual_status": [],
            "birth_control": [],
            "hormone_therapy": [],
            "fertility_treatments": [],
            "age_mentioned": False,
            "age_range": None,
            "confidence": 0.0,
        }


def create_medical_metadata_extractor(llm: LLM | None = None) -> MedicalMetadataExtractor:
    """
    Create a medical metadata extractor.

    Uses chunk-level extraction with paper_id caching (production pattern).
    """
    return MedicalMetadataExtractor(llm=llm)
