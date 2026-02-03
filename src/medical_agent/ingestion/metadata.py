
import logging
from typing import Any, Sequence

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.extractors import BaseExtractor, PydanticProgramExtractor
from llama_index.core.llms import ChatMessage, LLM, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.llms.azure_openai import AzureOpenAI

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


class MedicalMetadata(BaseModel):
    """
    Medical metadata extracted from research papers.

    Uses Pydantic BaseModel for native LlamaIndex integration with
    automatic serialization and structured output validation.
    """

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

    title: str | None = Field(
        default=None,
        description="Paper title if found in document (e.g., title page, header, or metadata)."
    )

    publication_year: int | None = Field(
        default=None,
        description="Publication year if found in document (e.g., '2017', '2023'). Extract as integer year only."
    )

    author: str | None = Field(
        default=None,
        description="Primary author(s) if found (e.g., 'Smith et al.' or 'Smith, J. and Jones, M.'). First author is sufficient."
    )

    doi: str | None = Field(
        default=None,
        description="Digital Object Identifier if found (e.g., '10.1234/example' or full URL). Extract identifier only."
    )


# Extraction prompt that emphasizes standardized terms
EXTRACTION_PROMPT = """You are extracting metadata from research papers for a women's health application.

CRITICAL RULES:
1. ONLY extract information that is EXPLICITLY MENTIONED in the text
2. DO NOT infer, extrapolate, or hallucinate
3. Use EXACT standardized terms from field descriptions

PAPER METADATA (extract if found):
- title: Paper title from title page or header
- publication_year: Year as integer (e.g., 2017, 2023)
- author: Primary author(s), first author sufficient (e.g., "Smith et al.")
- doi: Digital Object Identifier only, no URL (e.g., "10.1234/example")

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


class SimplifiedMedicalMetadataExtractor(BaseExtractor):
    """
    Simplified medical metadata extractor using LlamaIndex BaseExtractor.

    This replaces 650+ lines of custom code with ~100 lines using native
    LlamaIndex functionality:
    - Automatic caching per document (via ref_doc_id)
    - Structured output via Pydantic + LLM
    - Clean integration with IngestionPipeline
    """

    def __init__(self, llm: LLM | None = None, **kwargs):
        """Initialize the extractor with an LLM."""
        super().__init__(**kwargs)

        # Create LLM if not provided
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

        # Store LLM using object.__setattr__ to avoid Pydantic validation
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_cache", {})

        # Create prompt template with placeholder
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", EXTRACTION_PROMPT + "\n\n{text}")
        ])
        object.__setattr__(self, "_prompt_template", prompt_template)

        logger.info("Created simplified medical metadata extractor")

    def _extract_relevant_text(self, nodes: Sequence[BaseNode]) -> str:
        """Extract relevant text from abstract/methods sections."""
        parts = []

        for node in nodes[:5]:  # Check first 5 nodes
            content = node.get_content()
            section_type = node.metadata.get("section_type", "").lower()
            chunk_type = node.metadata.get("chunk_type", "").lower()

            # Prioritize abstract and methods
            if "abstract" in section_type or chunk_type == "abstract":
                parts.append(f"ABSTRACT:\n{content[:2000]}")
            elif "method" in section_type:
                parts.append(f"METHODS:\n{content[:2000]}")
            elif len(parts) < 3:
                parts.append(content[:1500])

        return "\n\n".join(parts) if parts else nodes[0].get_content()[:3000] if nodes else ""

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        """Extract medical metadata and stamp on all nodes."""
        if not nodes:
            return []

        try:
            # Check cache using document ID
            doc_id = nodes[0].ref_doc_id or "unknown"

            if doc_id in self._cache:
                logger.debug(f"Using cached metadata for document {doc_id}")
                cached_result = self._cache[doc_id]
            else:
                logger.info(f"Extracting metadata for document {doc_id} ({len(nodes)} nodes)")

                # Check if extraction is enabled
                if not getattr(settings, "metadata_extraction_enabled", True):
                    logger.warning("Metadata extraction is disabled in settings")
                    cached_result = self._empty_metadata()
                else:
                    # Extract relevant text
                    text = self._extract_relevant_text(nodes)

                    if not text.strip():
                        logger.warning(f"No text available for extraction (doc {doc_id})")
                        cached_result = self._empty_metadata()
                    else:
                        # Call LLM with structured output using prompt template
                        result = await self._llm.astructured_predict(
                            output_cls=MedicalMetadata,
                            prompt=self._prompt_template,
                            text=text,
                        )

                        # Convert Pydantic model to dict
                        cached_result = result.model_dump()

                        total_terms = sum(
                            len(v) for k, v in cached_result.items()
                            if isinstance(v, list)
                        )
                        logger.info(
                            f"Extracted {total_terms} terms for doc {doc_id}, "
                            f"confidence: {cached_result.get('confidence', 0.0):.2f}"
                        )

                # Cache result
                self._cache[doc_id] = cached_result

            # Return metadata for all nodes (same metadata for whole document)
            return [cached_result.copy() for _ in nodes]

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}", exc_info=True)
            return [self._empty_metadata() for _ in nodes]

    def _empty_metadata(self) -> dict[str, Any]:
        """Return empty metadata structure."""
        return {
            "title": None,
            "publication_year": None,
            "author": None,
            "doi": None,
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


def create_medical_metadata_extractor(
    llm: LLM | None = None,
) -> SimplifiedMedicalMetadataExtractor:
    """
    Create a simplified medical metadata extractor.

    This replaces:
    - MedicalMetadataExtractor (217 lines)
    - MetadataLLMClient (166 lines)
    - TermNormalizer (261 lines)
    - ExtractedMetadata dataclass

    With a single ~100 line class that uses:
    - LlamaIndex's structured_predict for Pydantic models
    - Native BaseExtractor caching
    - Prompt engineering for normalization (no regex needed)

    Args:
        llm: Optional LLM instance (creates Azure OpenAI if None)

    Returns:
        Configured extractor ready for use in IngestionPipeline
    """
    return SimplifiedMedicalMetadataExtractor(llm=llm)


def dict_to_medical_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert extracted metadata dict to storage format.

    LlamaIndex returns metadata as dicts. This function ensures
    consistent structure for storage in JSONB.

    Args:
        data: Raw metadata dict from LLMMetadataExtractor

    Returns:
        Cleaned metadata dict ready for storage
    """
    # LLMMetadataExtractor returns flat dict with field names as keys
    # Validate and clean
    return {
        "extracted_metadata": {
            "title": data.get("title"),
            "publication_year": data.get("publication_year"),
            "author": data.get("author"),
            "doi": data.get("doi"),
            "ethnicities": data.get("ethnicities", []),
            "diagnoses": data.get("diagnoses", []),
            "symptoms": data.get("symptoms", []),
            "menstrual_status": data.get("menstrual_status", []),
            "birth_control": data.get("birth_control", []),
            "hormone_therapy": data.get("hormone_therapy", []),
            "fertility_treatments": data.get("fertility_treatments", []),
            "age_mentioned": data.get("age_mentioned", False),
            "age_range": data.get("age_range"),
            "confidence": data.get("confidence", 0.0),
        }
    }
