"""LLM client for medical metadata extraction using structured output."""

import json
import logging

from openai import AsyncAzureOpenAI

from medical_agent.core.config import settings
from medical_agent.core.exceptions import LLMError

from .types import ExtractedMetadata

logger = logging.getLogger(__name__)


# JSON schema for structured output
METADATA_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "ethnicities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of ethnicity terms explicitly mentioned in the text",
        },
        "diagnoses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of hormone-related diagnoses explicitly mentioned",
        },
        "symptoms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of symptoms explicitly mentioned (discharge, odor, pain, etc.)",
        },
        "menstrual_status": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Menstrual cycle states mentioned (premenstrual, menstrual, etc.)",
        },
        "birth_control": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Birth control types mentioned (pill, IUD, etc.)",
        },
        "hormone_therapy": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hormone therapies mentioned (HRT, testosterone, etc.)",
        },
        "fertility_treatments": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fertility treatments mentioned (IVF, clomiphene, etc.)",
        },
        "age_mentioned": {
            "type": "boolean",
            "description": "Whether age is explicitly mentioned",
        },
        "age_range": {
            "type": ["string", "null"],
            "description": "Age range if mentioned (e.g., '25-35')",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score for extraction quality (0.0-1.0)",
        },
    },
    "required": [
        "ethnicities",
        "diagnoses",
        "symptoms",
        "menstrual_status",
        "birth_control",
        "hormone_therapy",
        "fertility_treatments",
        "age_mentioned",
        "age_range",
        "confidence",
    ],
    "additionalProperties": False,
}

class MetadataLLMClient:
    """LLM client for extracting medical metadata using structured output."""

    SYSTEM_PROMPT = """Extract ONLY explicitly mentioned medical terms from research papers.
Do not infer or hallucinate relevance. If not directly stated, don't extract it.
Return empty arrays for categories with no explicit mentions.

Categories: ethnicities, diagnoses, symptoms, menstrual status, birth control,
hormone therapy, fertility treatments, age. Provide a confidence score (0.0-1.0)."""

    def __init__(
        self,
        client: AsyncAzureOpenAI | None = None,
        model: str | None = None,
    ):
        """
        Initialize the metadata LLM client.

        Args:
            client: Optional Azure OpenAI client (creates one if None)
            model: Optional model name (uses settings.metadata_extraction_model if None)
        """
        self._client = client
        self.model = model or getattr(
            settings,
            "metadata_extraction_model",
            settings.azure_openai_deployment_name,
        )

    @property
    def client(self) -> AsyncAzureOpenAI:
        """Get or create the Azure OpenAI client."""
        if self._client is None:
            # Create async client directly using settings
            self._client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
            )
        return self._client

    async def extract_metadata(
        self,
        text: str,
        source_description: str = "research paper",
    ) -> ExtractedMetadata:
        """
        Extract medical metadata from text using GPT-4o.

        Args:
            text: Text to extract from (abstract + keywords + methods sections)
            source_description: Description of source for logging

        Returns:
            ExtractedMetadata with all categories

        Raises:
            LLMError: If extraction fails
        """
        logger.info(f"Extracting metadata from {source_description}")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extract medical metadata from this text:\n\n{text}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "medical_metadata",
                        "schema": METADATA_EXTRACTION_SCHEMA,
                        "strict": True,
                    },
                },
                temperature=0.0,  # Deterministic extraction
                max_tokens=1000,
            )

            # Parse structured output
            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from LLM")

            data = json.loads(content)

            # Create ExtractedMetadata from response
            metadata = ExtractedMetadata(
                ethnicities=data.get("ethnicities", []),
                diagnoses=data.get("diagnoses", []),
                symptoms=data.get("symptoms", []),
                menstrual_status=data.get("menstrual_status", []),
                birth_control=data.get("birth_control", []),
                hormone_therapy=data.get("hormone_therapy", []),
                fertility_treatments=data.get("fertility_treatments", []),
                age_mentioned=data.get("age_mentioned", False),
                age_range=data.get("age_range"),
                confidence=data.get("confidence", 0.0),
            )

            logger.info(
                f"Extracted metadata with {sum(len(getattr(metadata, field)) for field in ['ethnicities', 'diagnoses', 'symptoms', 'menstrual_status', 'birth_control', 'hormone_therapy', 'fertility_treatments'])} total terms, "
                f"confidence: {metadata.confidence:.2f}"
            )

            return metadata

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise LLMError(f"Invalid JSON from LLM: {e}")
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise LLMError(f"Metadata extraction failed: {e}")


# Global client instance
_llm_client: MetadataLLMClient | None = None


def get_metadata_llm_client() -> MetadataLLMClient:
    """Get or create the global metadata LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = MetadataLLMClient()
    return _llm_client
