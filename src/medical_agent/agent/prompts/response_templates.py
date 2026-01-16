"""
Response Templates by Risk Level

Provides structured templates for generating user-facing responses
that vary in tone and urgency based on the assessed risk level.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Final


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    NORMAL = "normal"
    MONITOR = "monitor"
    CONCERNING = "concerning"
    URGENT = "urgent"


@dataclass
class RiskLevelConfig:
    """Configuration for each risk level response."""
    level: RiskLevel
    emoji: str
    title: str
    tone: str
    urgency_phrase: str
    action_emphasis: str
    color_code: str  # For UI display


RISK_LEVEL_CONFIGS: dict[RiskLevel, RiskLevelConfig] = {
    RiskLevel.NORMAL: RiskLevelConfig(
        level=RiskLevel.NORMAL,
        emoji="✅",
        title="Your Results Look Normal",
        tone="reassuring and educational",
        urgency_phrase="",
        action_emphasis="Continue maintaining your vaginal health",
        color_code="#22c55e",  # Green
    ),
    RiskLevel.MONITOR: RiskLevelConfig(
        level=RiskLevel.MONITOR,
        emoji="📊",
        title="Some Things to Keep an Eye On",
        tone="informative and supportive",
        urgency_phrase="While not immediately concerning, ",
        action_emphasis="Consider tracking your symptoms over the next few days",
        color_code="#eab308",  # Yellow
    ),
    RiskLevel.CONCERNING: RiskLevelConfig(
        level=RiskLevel.CONCERNING,
        emoji="⚠️",
        title="We Recommend Professional Guidance",
        tone="caring but direct",
        urgency_phrase="Based on your results, ",
        action_emphasis="We recommend scheduling an appointment with a healthcare provider",
        color_code="#f97316",  # Orange
    ),
    RiskLevel.URGENT: RiskLevelConfig(
        level=RiskLevel.URGENT,
        emoji="🏥",
        title="Please Seek Medical Attention",
        tone="calm but serious",
        urgency_phrase="Your results suggest you should ",
        action_emphasis="Please contact a healthcare provider today",
        color_code="#ef4444",  # Red
    ),
}


# =============================================================================
# Response Templates by Risk Level
# =============================================================================

NORMAL_RESPONSE_TEMPLATE: Final[str] = """## ✅ Your Results Look Normal

**Your pH Level:** {ph_value}

Great news! Your vaginal pH of {ph_value} falls within the normal healthy range of 3.8-4.5. This indicates a balanced vaginal environment.

### What the Research Tells Us

{research_insights}

### Personalized Insights

{personalized_insights}

### Maintaining Your Vaginal Health

{wellness_tips}

---

{citations}

{disclaimer}"""


MONITOR_RESPONSE_TEMPLATE: Final[str] = """## 📊 Some Things to Keep an Eye On

**Your pH Level:** {ph_value}

Your vaginal pH reading of {ph_value} is {ph_assessment}. While this isn't cause for immediate concern, it's worth monitoring.

### What the Research Tells Us

{research_insights}

### What This Might Mean for You

{personalized_insights}

### Recommended Next Steps

{monitoring_steps}

### When to Seek Care

Consider contacting a healthcare provider if you notice:
- Symptoms persisting more than a few days
- New or worsening symptoms
- Any unusual discharge, odor, or discomfort

---

{citations}

{disclaimer}"""


CONCERNING_RESPONSE_TEMPLATE: Final[str] = """## ⚠️ We Recommend Professional Guidance

**Your pH Level:** {ph_value}

Your vaginal pH reading of {ph_value} combined with your reported symptoms suggests it would be beneficial to consult with a healthcare provider.

### Why We Recommend Professional Consultation

{risk_explanation}

### What the Research Tells Us

{research_insights}

### Based on Your Profile

{personalized_insights}

### Recommended Actions

1. **Schedule an appointment** with your gynecologist or primary care provider
2. **Document your symptoms** - note when they started and any changes
3. **Avoid self-treatment** until you've consulted a professional

### What to Tell Your Doctor

{doctor_discussion_points}

---

{citations}

{disclaimer}"""


URGENT_RESPONSE_TEMPLATE: Final[str] = """## 🏥 Please Seek Medical Attention

**Your pH Level:** {ph_value}

**We strongly recommend contacting a healthcare provider today.**

Your test results and symptoms indicate that professional medical evaluation is important.

### Why This Needs Attention

{urgent_explanation}

### Immediate Steps

1. **Contact your healthcare provider** or visit an urgent care clinic today
2. If you experience severe symptoms (fever, severe pain, heavy bleeding), consider emergency care
3. Do not delay seeking care

### What to Tell Medical Staff

{emergency_discussion_points}

### While You Wait for Your Appointment

{interim_guidance}

---

{citations}

{disclaimer}"""


# =============================================================================
# Template Selection and Formatting
# =============================================================================

RESPONSE_TEMPLATES: dict[RiskLevel, str] = {
    RiskLevel.NORMAL: NORMAL_RESPONSE_TEMPLATE,
    RiskLevel.MONITOR: MONITOR_RESPONSE_TEMPLATE,
    RiskLevel.CONCERNING: CONCERNING_RESPONSE_TEMPLATE,
    RiskLevel.URGENT: URGENT_RESPONSE_TEMPLATE,
}


def get_response_template(risk_level: RiskLevel | str) -> str:
    """Get the appropriate response template for a risk level."""
    if isinstance(risk_level, str):
        risk_level = RiskLevel(risk_level.lower())
    return RESPONSE_TEMPLATES.get(risk_level, NORMAL_RESPONSE_TEMPLATE)


def get_risk_config(risk_level: RiskLevel | str) -> RiskLevelConfig:
    """Get the configuration for a risk level."""
    if isinstance(risk_level, str):
        risk_level = RiskLevel(risk_level.lower())
    return RISK_LEVEL_CONFIGS.get(risk_level, RISK_LEVEL_CONFIGS[RiskLevel.NORMAL])


# =============================================================================
# Wellness Tips by Risk Level
# =============================================================================

WELLNESS_TIPS_NORMAL: Final[list[str]] = [
    "Continue with your current hygiene practices",
    "Wear breathable, cotton underwear",
    "Stay hydrated and maintain a balanced diet",
    "Avoid douching or using harsh soaps in the vaginal area",
    "Consider probiotic foods that support vaginal health",
]

MONITORING_STEPS: Final[list[str]] = [
    "Test your pH again in 2-3 days to track any changes",
    "Keep note of any new symptoms that develop",
    "Maintain good hygiene practices",
    "Avoid potential irritants (scented products, tight clothing)",
    "Stay hydrated and get adequate rest",
]

DOCTOR_DISCUSSION_POINTS: Final[list[str]] = [
    "Your recent pH readings and any trends",
    "All symptoms you're experiencing, including duration",
    "Any recent changes in sexual activity, hygiene products, or medications",
    "Relevant medical history",
    "Any over-the-counter treatments you've tried",
]

EMERGENCY_DISCUSSION_POINTS: Final[list[str]] = [
    "Your pH reading and when it was taken",
    "All current symptoms and their severity",
    "When symptoms started and if they're worsening",
    "Any fever, severe pain, or unusual bleeding",
    "Current medications and recent sexual activity",
    "Pregnancy status if applicable",
]


# =============================================================================
# pH Assessment Phrases
# =============================================================================

PH_ASSESSMENT_PHRASES: dict[str, str] = {
    "normal": "within the normal healthy range",
    "slightly_elevated": "slightly above the normal range",
    "elevated": "elevated above the normal range",
    "highly_elevated": "significantly elevated and outside the normal range",
}


def get_ph_assessment_phrase(ph_category: str) -> str:
    """Get a human-readable assessment phrase for a pH category."""
    return PH_ASSESSMENT_PHRASES.get(
        ph_category.lower(),
        "requires evaluation"
    )

