"""
Metadata LLM Client - Utility Component

LLM client for metadata extraction operations.
Used by MedicalMetadataExtractor (which is wrapped by LlamaIndex pipeline).
"""

import json
import logging
from typing import Any

from openai import AsyncAzureOpenAI

from medical_agent.core.config import settings
from medical_agent.core.exceptions import LLMError

from .types import ExtractedMetadata, TableMetadata

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

TABLE_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "integer"},
                    "summary": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["table_id", "summary", "confidence"],
            },
        },
    },
    "required": ["summaries"],
    "additionalProperties": False,
}


class MetadataLLMClient:
    """
    LLM client for extracting medical metadata using GPT-4o.

    Uses Azure OpenAI with structured output to ensure consistent extraction.
    Only extracts explicitly mentioned terms - no hallucinations.
    """

    SYSTEM_PROMPT = """You are a medical research paper metadata extractor.

Your task is to extract ONLY explicitly mentioned medical terms from research papers.

CRITICAL RULES:
1. Only extract terms that are EXPLICITLY MENTIONED in the source text
2. NEVER infer or hallucinate relevance - if it's not directly stated, don't extract it
3. Return empty arrays for categories with no explicit mentions
4. Be precise - only extract medical terms that are clearly present in the text

Categories to extract:
- Ethnicities: African / Black, Asian, Caucasian, Hispanic / Latina, etc.
- Diagnoses: PCOS, Endometriosis, Bacterial vaginosis, etc.
- Symptoms: Vaginal discharge, odor, itching, pain, etc.
- Menstrual status: premenstrual, menstrual, postmenstrual, etc.
- Birth control: pills, IUD, implant, etc.
- Hormone therapy: HRT, testosterone, estrogen, etc.
- Fertility treatments: IVF, clomiphene, ovulation induction, etc.
- Age: note if age or age range is mentioned

If a category is not explicitly discussed in the text, return an empty array.
Provide a confidence score (0.0-1.0) based on how clearly the terms were stated."""

    TABLE_SUMMARY_PROMPT = """You are a medical research table summarizer.

Your task is to create ONE-SENTENCE summaries for tables in medical research papers.

For each table, describe:
1. What variables or measurements are included
2. The main focus or comparison being shown

Keep summaries concise (one sentence, max 20 words).
Examples:
- "Comparison of vaginal pH levels across different ethnic groups and age ranges."
- "Prevalence of bacterial vaginosis symptoms stratified by birth control method."

Be specific about what the table contains, not what it might show or conclude."""

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

    async def summarize_tables(
        self,
        tables: list[dict[str, Any]],
    ) -> dict[int, TableMetadata]:
        """
        Generate one-sentence summaries for tables.

        Args:
            tables: List of table dictionaries with table_id and markdown content

        Returns:
            Dict mapping table_id to TableMetadata

        Raises:
            LLMError: If summarization fails
        """
        if not tables:
            return {}

        logger.info(f"Summarizing {len(tables)} tables")

        try:
            # Format tables for prompt
            tables_text = "\n\n".join(
                [
                    f"Table {t['table_id']}:\n{t.get('markdown', t.get('content', ''))}"
                    for t in tables
                ]
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.TABLE_SUMMARY_PROMPT},
                    {
                        "role": "user",
                        "content": f"Summarize these tables:\n\n{tables_text}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "table_summaries",
                        "schema": TABLE_SUMMARY_SCHEMA,
                        "strict": True,
                    },
                },
                temperature=0.0,
                max_tokens=500,
            )

            # Parse structured output
            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from LLM")

            data = json.loads(content)

            # Create TableMetadata dict
            summaries = {}
            for item in data.get("summaries", []):
                table_id = item["table_id"]
                summaries[table_id] = TableMetadata(
                    table_id=table_id,
                    summary=item["summary"],
                    confidence=item.get("confidence", 0.0),
                )

            logger.info(f"Generated {len(summaries)} table summaries")
            return summaries

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise LLMError(f"Invalid JSON from LLM: {e}")
        except Exception as e:
            logger.error(f"Table summarization failed: {e}")
            raise LLMError(f"Table summarization failed: {e}")


# Global client instance
_llm_client: MetadataLLMClient | None = None


def get_metadata_llm_client() -> MetadataLLMClient:
    """Get or create the global metadata LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = MetadataLLMClient()
    return _llm_client
