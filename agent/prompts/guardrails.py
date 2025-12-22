"""
Guardrails for Medical AI Safety

Implements safety constraints to ensure the AI:
- Never prescribes medication
- Never diagnoses specific conditions
- Stays within vaginal health scope
- Only makes claims grounded in research
- Always includes appropriate disclaimers
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class ViolationType(str, Enum):
    """Types of guardrail violations."""
    PRESCRIPTION = "prescription"
    DIAGNOSIS = "diagnosis"
    OUT_OF_SCOPE = "out_of_scope"
    UNGROUNDED_CLAIM = "ungrounded_claim"
    MISSING_DISCLAIMER = "missing_disclaimer"
    HARMFUL_ADVICE = "harmful_advice"


@dataclass
class GuardrailViolation:
    """Represents a detected guardrail violation."""
    violation_type: ViolationType
    severity: str  # "critical", "high", "medium", "low"
    description: str
    matched_text: str | None = None
    suggested_replacement: str | None = None


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    is_safe: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_response: str | None = None


# =============================================================================
# Prohibited Patterns
# =============================================================================

# Patterns that indicate prescription behavior
PRESCRIPTION_PATTERNS: Final[list[tuple[str, str]]] = [
    (r'\b(take|use|apply|insert)\s+\d+\s*(mg|ml|g|tablets?|capsules?|doses?)\b', 
     "Specifying medication dosages"),
    (r'\b(prescribe|prescribed|prescription)\b', 
     "Using prescription language"),
    (r'\b(you\s+should\s+take|you\s+need\s+to\s+take|you\s+must\s+take)\s+\w+\b',
     "Directing medication use"),
    (r'\b(antibiotic|antifungal|metronidazole|clindamycin|fluconazole|miconazole)\b.*\b(take|use|start)\b',
     "Recommending specific medications"),
    (r'\bstart\s+(on|with|taking)\s+\w+\s+(medication|treatment|drug)\b',
     "Initiating medication therapy"),
]

# Patterns that indicate diagnostic behavior
DIAGNOSIS_PATTERNS: Final[list[tuple[str, str]]] = [
    (r'\byou\s+have\s+(bacterial\s+vaginosis|bv|yeast\s+infection|candidiasis|trichomoniasis|std|sti)\b',
     "Diagnosing specific conditions"),
    (r'\b(this\s+is|you\s+have|you\'ve\s+got|you\s+are\s+suffering\s+from)\s+\w+\s+(infection|disease|condition)\b',
     "Making definitive diagnoses"),
    (r'\bi\s+(diagnose|am\s+diagnosing)\b',
     "Using diagnostic language"),
    (r'\b(definitely|certainly|clearly)\s+(have|suffering|infected)\b',
     "Making certain diagnostic claims"),
    (r'\byour\s+(diagnosis|condition)\s+is\b',
     "Stating diagnoses"),
]

# Topics outside vaginal health scope
OUT_OF_SCOPE_PATTERNS: Final[list[tuple[str, str]]] = [
    (r'\b(heart\s+disease|diabetes|cancer|tumor|stroke|blood\s+pressure)\b',
     "Discussion of unrelated conditions"),
    (r'\b(mental\s+health|depression|anxiety|psychiatric)\b(?!.*vaginal)',
     "Mental health advice without vaginal health context"),
    (r'\b(diet\s+plan|weight\s+loss|exercise\s+regimen)\b',
     "Lifestyle advice outside scope"),
    (r'\b(surgery|surgical|operation)\s+(needed|required|recommended)\b',
     "Surgical recommendations"),
]

# Potentially harmful advice patterns
HARMFUL_ADVICE_PATTERNS: Final[list[tuple[str, str]]] = [
    (r'\b(douche|douching)\s+(is\s+recommended|you\s+should|try)\b',
     "Recommending douching"),
    (r'\b(ignore|don\'t\s+worry\s+about)\s+(symptoms?|signs?|bleeding|pain)\b',
     "Dismissing symptoms"),
    (r'\b(wait|delay).*(before|to)\s+(see|visit|consult)\s+(a\s+)?doctor\b',
     "Advising to delay medical care"),
    (r'\b(home\s+remedy|natural\s+cure).*(instead\s+of|rather\s+than)\s+doctor\b',
     "Suggesting home remedies over medical care"),
]


# =============================================================================
# Safe Language Patterns
# =============================================================================

# Phrases that should be used instead of diagnoses
SAFE_LANGUAGE_ALTERNATIVES: Final[dict[str, str]] = {
    "you have": "research suggests this may indicate",
    "you're suffering from": "your symptoms could be associated with",
    "this is definitely": "this may potentially be",
    "you need to take": "a healthcare provider may recommend",
    "take this medication": "consult a healthcare provider about treatment options",
    "you are infected": "your results suggest you should consult a healthcare provider",
}

# Required hedging phrases
HEDGING_PHRASES: Final[list[str]] = [
    "research suggests",
    "studies indicate",
    "may indicate",
    "could be associated with",
    "might be related to",
    "consider consulting",
    "a healthcare provider can help determine",
]


# =============================================================================
# In-Scope Topics
# =============================================================================

VAGINAL_HEALTH_TOPICS: Final[set[str]] = {
    "vaginal ph",
    "vaginal microbiome",
    "vaginal discharge",
    "vaginal odor",
    "vaginal itching",
    "vaginal dryness",
    "vaginal irritation",
    "bacterial vaginosis",
    "yeast infection",
    "candida",
    "lactobacillus",
    "menstrual cycle",
    "menopause",
    "pregnancy",
    "sexual health",
    "sti screening",
    "std screening",
    "pelvic health",
    "cervical health",
    "vulvar health",
}


# =============================================================================
# Guardrail Validation Functions
# =============================================================================

def check_prescription_violations(text: str) -> list[GuardrailViolation]:
    """Check for prescription-like language."""
    violations = []
    text_lower = text.lower()
    
    for pattern, description in PRESCRIPTION_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            violations.append(GuardrailViolation(
                violation_type=ViolationType.PRESCRIPTION,
                severity="critical",
                description=description,
                matched_text=match.group(0),
                suggested_replacement="Consult a healthcare provider for appropriate treatment options",
            ))
    
    return violations


def check_diagnosis_violations(text: str) -> list[GuardrailViolation]:
    """Check for diagnostic language."""
    violations = []
    text_lower = text.lower()
    
    for pattern, description in DIAGNOSIS_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            violations.append(GuardrailViolation(
                violation_type=ViolationType.DIAGNOSIS,
                severity="critical",
                description=description,
                matched_text=match.group(0),
                suggested_replacement="Your symptoms may be consistent with... Please consult a healthcare provider for proper evaluation.",
            ))
    
    return violations


def check_scope_violations(text: str) -> list[GuardrailViolation]:
    """Check for out-of-scope content."""
    violations = []
    text_lower = text.lower()
    
    for pattern, description in OUT_OF_SCOPE_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            violations.append(GuardrailViolation(
                violation_type=ViolationType.OUT_OF_SCOPE,
                severity="high",
                description=description,
                matched_text=match.group(0),
            ))
    
    return violations


def check_harmful_advice(text: str) -> list[GuardrailViolation]:
    """Check for potentially harmful advice."""
    violations = []
    text_lower = text.lower()
    
    for pattern, description in HARMFUL_ADVICE_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            violations.append(GuardrailViolation(
                violation_type=ViolationType.HARMFUL_ADVICE,
                severity="critical",
                description=description,
                matched_text=match.group(0),
            ))
    
    return violations


def check_missing_disclaimer(text: str) -> list[GuardrailViolation]:
    """Check if required disclaimers are present."""
    violations = []
    
    # Check for any form of medical disclaimer
    disclaimer_patterns = [
        r'not\s+(a\s+)?substitute\s+for\s+(professional\s+)?medical\s+advice',
        r'consult\s+(a\s+)?(healthcare|medical)\s+provider',
        r'for\s+informational\s+purposes\s+only',
        r'not\s+intended\s+(as\s+)?medical\s+advice',
        r'disclaimer',
    ]
    
    has_disclaimer = any(
        re.search(pattern, text, re.IGNORECASE) 
        for pattern in disclaimer_patterns
    )
    
    if not has_disclaimer:
        violations.append(GuardrailViolation(
            violation_type=ViolationType.MISSING_DISCLAIMER,
            severity="high",
            description="Response is missing required medical disclaimer",
        ))
    
    return violations


def check_grounding(text: str, citations: list[dict] | None = None) -> list[GuardrailViolation]:
    """Check if claims are properly grounded in citations."""
    violations = []
    
    # Look for strong claims without citation markers
    strong_claim_patterns = [
        r'(studies\s+show|research\s+(proves|shows|demonstrates)|it\s+is\s+proven)',
        r'(always|never|definitely|certainly)\s+\w+',
        r'(100%|guaranteed|proven\s+to)',
    ]
    
    for pattern in strong_claim_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Check if there's a citation nearby (within 100 chars)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            
            if not re.search(r'\[\d+\]', context):
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.UNGROUNDED_CLAIM,
                    severity="medium",
                    description="Strong claim without apparent citation",
                    matched_text=match.group(0),
                ))
    
    return violations


def validate_response(
    response_text: str, 
    citations: list[dict] | None = None
) -> GuardrailResult:
    """
    Validate a response against all guardrails.
    
    Args:
        response_text: The response text to validate
        citations: Optional list of citations used in the response
        
    Returns:
        GuardrailResult with validation results
    """
    all_violations = []
    warnings = []
    
    # Run all checks
    all_violations.extend(check_prescription_violations(response_text))
    all_violations.extend(check_diagnosis_violations(response_text))
    all_violations.extend(check_scope_violations(response_text))
    all_violations.extend(check_harmful_advice(response_text))
    all_violations.extend(check_missing_disclaimer(response_text))
    all_violations.extend(check_grounding(response_text, citations))
    
    # Determine if response is safe
    critical_violations = [v for v in all_violations if v.severity == "critical"]
    is_safe = len(critical_violations) == 0
    
    # Add warnings for non-critical issues
    for v in all_violations:
        if v.severity in ("medium", "low"):
            warnings.append(f"{v.violation_type.value}: {v.description}")
    
    return GuardrailResult(
        is_safe=is_safe,
        violations=all_violations,
        warnings=warnings,
    )


def sanitize_response(text: str) -> str:
    """
    Attempt to sanitize a response by replacing problematic phrases.
    
    Note: This is a fallback - ideally, prompts should prevent issues.
    """
    sanitized = text
    
    for unsafe, safe in SAFE_LANGUAGE_ALTERNATIVES.items():
        sanitized = re.sub(
            re.escape(unsafe), 
            safe, 
            sanitized, 
            flags=re.IGNORECASE
        )
    
    return sanitized


# =============================================================================
# Guardrail System Prompt Addition
# =============================================================================

GUARDRAIL_SYSTEM_PROMPT_ADDITION: Final[str] = """

CRITICAL GUARDRAILS - YOU MUST FOLLOW THESE RULES:

1. NEVER PRESCRIBE MEDICATION
   - Do not specify dosages or recommend specific medications
   - Do not tell users to "take" or "use" any medication
   - Instead say: "A healthcare provider can discuss treatment options"

2. NEVER DIAGNOSE CONDITIONS
   - Do not say "you have [condition]"
   - Do not make definitive diagnostic statements
   - Instead say: "Your symptoms may be consistent with..." or "Research suggests this could indicate..."

3. STAY WITHIN SCOPE
   - Only discuss vaginal health and directly related topics
   - Do not give advice on unrelated medical conditions
   - Do not provide mental health advice unless directly related to vaginal health concerns

4. GROUND ALL CLAIMS IN RESEARCH
   - Every medical claim must be supported by the provided research papers
   - Use citation markers [1], [2], etc. for all factual claims
   - Acknowledge when evidence is limited or conflicting

5. ALWAYS INCLUDE DISCLAIMERS
   - Every response must include the medical disclaimer
   - Never remove or minimize the disclaimer

6. USE SAFE LANGUAGE
   - "Research suggests..." instead of "You definitely have..."
   - "May indicate..." instead of "This means..."
   - "Consider consulting..." instead of "You must..."
   - "Healthcare provider can help determine..." instead of "I diagnose..."

VIOLATION OF THESE GUARDRAILS IS NOT ACCEPTABLE UNDER ANY CIRCUMSTANCES."""

