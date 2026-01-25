"""Normalizes extracted medical terms to standardized dropdown values."""

import logging
import re
from typing import Any

from .types import ExtractedMetadata

logger = logging.getLogger(__name__)


# Normalization mappings: extracted term patterns -> dropdown value
ETHNICITY_MAPPINGS = {
    # African / Black
    r"\b(african|black|sub-?saharan)\b": "African / Black",
    # Asian
    r"\b(asian|east asian|south asian|southeast asian|chinese|japanese|korean|indian|pakistani|vietnamese)\b": "Asian",
    # Caucasian
    r"\b(caucasian|white|european|euro)\b": "Caucasian",
    # Hispanic / Latina
    r"\b(hispanic|latina?|mexican|central american|south american)\b": "Hispanic / Latina",
    # Middle Eastern
    r"\b(middle eastern|arab|persian|iranian)\b": "Middle Eastern",
    # Mixed
    r"\b(mixed|multiracial|biracial)\b": "Mixed",
    # Native American / Indigenous
    r"\b(native american|indigenous|first nations|aboriginal)\b": "Native American / Indigenous",
    # North African
    r"\b(north african|maghreb|moroccan|algerian|tunisian|libyan|egyptian)\b": "North African",
    # Pacific Islander
    r"\b(pacific islander|polynesian|melanesian|micronesian|hawaiian|samoan|tongan|fijian)\b": "Pacific Islander",
    # South Asian
    r"\b(south asian|indian subcontinent|bangladeshi|sri lankan|nepali)\b": "South Asian",
    # Southeast Asian
    r"\b(southeast asian|filipino|thai|indonesian|malaysian|burmese|cambodian)\b": "Southeast Asian",
}

DIAGNOSES_MAPPINGS = {
    # Adenomyosis
    r"\b(adenomyosis)\b": "Adenomyosis",
    # Endometriosis
    r"\b(endometriosis)\b": "Endometriosis",
    # Bacterial vaginosis
    r"\b(bacterial vaginosis|bv|gardnerella)\b": "Bacterial vaginosis",
    # Yeast infection
    r"\b(yeast infection|candidiasis|candida|thrush)\b": "Yeast infection",
    # Sexually transmitted infection
    r"\b(sti|std|sexually transmitted|chlamydia|gonorrhea|trichomoniasis|herpes|hpv|syphilis)\b": "Sexually transmitted infection",
    # Polycystic ovary syndrome (PCOS)
    r"\b(pcos|polycystic ovary|polycystic ovarian)\b": "Polycystic ovary syndrome (PCOS)",
    # Premature ovarian insufficiency
    r"\b(premature ovarian|poi|early menopause)\b": "Premature ovarian insufficiency",
    # Thyroid disorder
    r"\b(thyroid|hypothyroid|hyperthyroid|hashimoto|graves)\b": "Thyroid disorder",
    # Fibroids (uterine myomas)
    r"\b(fibroid|myoma|leiomyoma|uterine mass)\b": "Fibroids (uterine myomas)",
    # Ovarian cysts
    r"\b(ovarian cyst|cystic ovary)\b": "Ovarian cysts",
    # Pelvic inflammatory disease
    r"\b(pelvic inflammatory|pid)\b": "Pelvic inflammatory disease",
}

SYMPTOMS_MAPPINGS = {
    # Discharge types
    r"\b(white discharge|creamy|milky)\b": "Creamy",
    r"\b(clear discharge|watery|transparent)\b": "Clear",
    r"\b(yellow discharge|yellowish)\b": "Yellow",
    r"\b(green discharge|greenish)\b": "Green",
    r"\b(gray discharge|grey|grayish|greyish)\b": "Gray",
    r"\b(pink discharge|pinkish|spotting)\b": "Pink",
    r"\b(brown discharge|brownish)\b": "Brown",
    # Odor
    r"\b(malodor|odor|smell|fishy|foul|unpleasant smell)\b": "Vaginal Odor",
    # Vulva/Vagina symptoms
    r"\b(itch|itching|pruritus)\b": "Itchy",
    r"\b(burning|burn)\b": "Burning",
    r"\b(pain|painful|dyspareunia|pelvic pain)\b": "Pelvic Pain",
    r"\b(swelling|swollen|edema)\b": "Swelling",
    r"\b(redness|red|erythema)\b": "Redness",
    r"\b(dryness|dry|atrophy)\b": "Vaginal Dryness",
    # Urine symptoms
    r"\b(frequent urination|polyuria|urgency)\b": "Frequent Urination",
    r"\b(painful urination|dysuria|burning urination)\b": "Painful Urination",
}

MENSTRUAL_STATUS_MAPPINGS = {
    r"\b(premenstrual|pre-?menstrual|before period|pms)\b": "Premenstrual",
    r"\b(menstrual|menstruation|period|menses|during period)\b": "Menstrual",
    r"\b(postmenstrual|post-?menstrual|after period)\b": "Postmenstrual",
    r"\b(ovulation|ovulatory|mid-?cycle)\b": "Ovulation",
    r"\b(luteal phase|luteal)\b": "Luteal Phase",
    r"\b(follicular phase|follicular)\b": "Follicular Phase",
}

BIRTH_CONTROL_MAPPINGS = {
    r"\b(pill|oral contraceptive|ocp|combined pill|birth control pill)\b": "Pill",
    r"\b(iud|intrauterine|mirena|paragard|copper iud|hormonal iud)\b": "IUD",
    r"\b(implant|nexplanon|implanon)\b": "Implant",
    r"\b(patch|contraceptive patch|ortho evra)\b": "Patch",
    r"\b(ring|vaginal ring|nuvaring)\b": "Ring",
    r"\b(injection|shot|depo|depo-?provera)\b": "Injection",
    r"\b(condom)\b": "Condom",
    r"\b(diaphragm)\b": "Diaphragm",
    r"\b(sterilization|tubal ligation|vasectomy)\b": "Sterilization",
}

HORMONE_THERAPY_MAPPINGS = {
    r"\b(hrt|hormone replacement|estrogen therapy|hormone therapy)\b": "HRT",
    r"\b(testosterone|androgen)\b": "Testosterone",
    r"\b(progesterone|progestin)\b": "Progesterone",
    r"\b(estrogen|estradiol|premarin)\b": "Estrogen",
    r"\b(thyroid medication|levothyroxine|synthroid)\b": "Thyroid Medication",
}

FERTILITY_TREATMENT_MAPPINGS = {
    r"\b(ivf|in vitro|assisted reproduction)\b": "IVF",
    r"\b(iui|intrauterine insemination|artificial insemination)\b": "IUI",
    r"\b(clomid|clomiphene)\b": "Clomiphene",
    r"\b(letrozole|femara)\b": "Letrozole",
    r"\b(gonadotropin|fsh|lh|menopur|gonal-?f)\b": "Gonadotropins",
    r"\b(ovulation induction)\b": "Ovulation Induction",
}


class TermNormalizer:
    """
    Normalizes extracted medical terms to standardized dropdown values.

    Maps synonyms, abbreviations, and variations to consistent terminology
    matching the dropdown options from user_input.pdf.
    """

    def __init__(self):
        """Initialize the term normalizer with mapping rules."""
        self.ethnicity_patterns = self._compile_patterns(ETHNICITY_MAPPINGS)
        self.diagnoses_patterns = self._compile_patterns(DIAGNOSES_MAPPINGS)
        self.symptoms_patterns = self._compile_patterns(SYMPTOMS_MAPPINGS)
        self.menstrual_patterns = self._compile_patterns(MENSTRUAL_STATUS_MAPPINGS)
        self.birth_control_patterns = self._compile_patterns(BIRTH_CONTROL_MAPPINGS)
        self.hormone_therapy_patterns = self._compile_patterns(HORMONE_THERAPY_MAPPINGS)
        self.fertility_patterns = self._compile_patterns(FERTILITY_TREATMENT_MAPPINGS)

    def _compile_patterns(
        self, mappings: dict[str, str]
    ) -> list[tuple[re.Pattern, str]]:
        """Compile regex patterns for efficient matching."""
        return [(re.compile(pattern, re.IGNORECASE), value) for pattern, value in mappings.items()]

    def _normalize_term(
        self,
        term: str,
        patterns: list[tuple[re.Pattern, str]],
    ) -> str | None:
        """
        Normalize a single term using pattern matching.

        Args:
            term: Raw extracted term
            patterns: List of (compiled_pattern, normalized_value) tuples

        Returns:
            Normalized term if match found, None otherwise
        """
        term_lower = term.lower().strip()

        for pattern, normalized_value in patterns:
            if pattern.search(term_lower):
                return normalized_value

        # No match found - return original term for manual review
        logger.debug(f"No normalization found for term: {term}")
        return None

    def _normalize_list(
        self,
        terms: list[str],
        patterns: list[tuple[re.Pattern, str]],
    ) -> list[str]:
        """
        Normalize a list of terms, removing duplicates.

        Args:
            terms: List of raw extracted terms
            patterns: List of (compiled_pattern, normalized_value) tuples

        Returns:
            List of unique normalized terms
        """
        normalized = set()

        for term in terms:
            norm_term = self._normalize_term(term, patterns)
            if norm_term:
                normalized.add(norm_term)

        return sorted(list(normalized))

    def normalize(self, metadata: ExtractedMetadata) -> ExtractedMetadata:
        """
        Normalize all terms in extracted metadata.

        Args:
            metadata: Raw extracted metadata

        Returns:
            Normalized metadata with standardized dropdown values
        """
        logger.info("Normalizing extracted metadata terms")

        normalized = ExtractedMetadata(
            ethnicities=self._normalize_list(metadata.ethnicities, self.ethnicity_patterns),
            diagnoses=self._normalize_list(metadata.diagnoses, self.diagnoses_patterns),
            symptoms=self._normalize_list(metadata.symptoms, self.symptoms_patterns),
            menstrual_status=self._normalize_list(metadata.menstrual_status, self.menstrual_patterns),
            birth_control=self._normalize_list(metadata.birth_control, self.birth_control_patterns),
            hormone_therapy=self._normalize_list(metadata.hormone_therapy, self.hormone_therapy_patterns),
            fertility_treatments=self._normalize_list(metadata.fertility_treatments, self.fertility_patterns),
            age_mentioned=metadata.age_mentioned,
            age_range=metadata.age_range,
            table_summaries=metadata.table_summaries,
            confidence=metadata.confidence,
        )

        logger.info(
            f"Normalized metadata: {len(normalized.ethnicities)} ethnicities, "
            f"{len(normalized.diagnoses)} diagnoses, {len(normalized.symptoms)} symptoms, "
            f"{len(normalized.menstrual_status)} menstrual states, "
            f"{len(normalized.birth_control)} birth control, "
            f"{len(normalized.hormone_therapy)} hormone therapies, "
            f"{len(normalized.fertility_treatments)} fertility treatments"
        )

        return normalized


# Global normalizer instance
_normalizer: TermNormalizer | None = None


def get_term_normalizer() -> TermNormalizer:
    """Get or create the global term normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = TermNormalizer()
    return _normalizer
