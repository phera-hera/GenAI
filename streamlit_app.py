"""Medical RAG Streamlit Application"""

import html
import logging
import re
import uuid
from typing import Any

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="pHera · pH Analysis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styles ──
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    .stApp { font-family: 'Inter', sans-serif; }

    h1, h2, h3, h4 { color: #f1f5f9; }

    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .app-header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; }
    .app-header p { color: #94a3b8; font-size: 0.95rem; margin-top: 0; }

    .ph-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Hide form border — inner cards handle visuals */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
    }

    /* ── Inline citation markers [1], [2] etc. ── */
    .cite-ref {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #818cf8;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.72em;
        vertical-align: super;
        line-height: 1;
        margin: 0 1px;
        padding: 1px 5px;
        background: rgba(99, 102, 241, 0.12);
        border-radius: 4px;
    }
    .cite-ref:hover { color: #a5b4fc; background: rgba(99, 102, 241, 0.22); }

    .cite-ref .cite-tip {
        visibility: hidden;
        width: min(360px, 85vw);
        background: #1a1f2e;
        color: #cbd5e1;
        text-align: left;
        border-radius: 10px;
        padding: 0;
        border: 1px solid rgba(99, 102, 241, 0.25);
        position: absolute;
        z-index: 1000;
        bottom: calc(100% + 10px);
        left: 50%;
        transform: translateX(-50%) scale(0.95);
        opacity: 0;
        transition: opacity 0.15s ease, transform 0.15s ease;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        font-size: 0.85rem;
        line-height: 1.5;
        overflow: hidden;
        pointer-events: none;
    }
    .cite-ref:hover .cite-tip {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) scale(1);
        pointer-events: auto;
    }
    .cite-tip-title {
        padding: 10px 14px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(99, 102, 241, 0.04));
        border-bottom: 1px solid rgba(99, 102, 241, 0.12);
        font-weight: 600;
        color: #e2e8f0;
        font-size: 0.84rem;
    }
    .cite-tip-body {
        padding: 10px 14px;
        color: #94a3b8;
        font-size: 0.8rem;
        line-height: 1.6;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .cite-ref .cite-tip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: rgba(99, 102, 241, 0.25) transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/api/v1/query"

# ── Session State ──
for key, default in {
    "page": "form",
    "session_id": str(uuid.uuid4()),
    "ph_value": None,
    "health_profile": {},
    "chat_history": [],
    "first_response": None,
    "first_citations": None,
    "disclaimers": None,
    "processing_time_ms": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API ──
def call_medical_rag_api(
    ph_value: float,
    health_profile: dict[str, Any],
    user_message: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        payload = {
            "ph_value": ph_value,
            "user_message": user_message,
            "session_id": session_id,
            "age": health_profile.get("age"),
            "diagnoses": health_profile.get("diagnoses", []),
            "ethnic_backgrounds": health_profile.get("ethnic_backgrounds", []),
            "menstrual_cycle": health_profile.get("menstrual_cycle"),
            "birth_control": health_profile.get("birth_control"),
            "hormone_therapy": health_profile.get("hormone_therapy", []),
            "hrt": health_profile.get("hrt", []),
            "fertility_journey": health_profile.get("fertility_journey"),
            "symptoms": {
                "discharge": health_profile.get("symptoms", {}).get("discharge", []),
                "vulva_vagina": health_profile.get("symptoms", {}).get("vulva_vagina", []),
                "smell": health_profile.get("symptoms", {}).get("smell", []),
                "urine": health_profile.get("symptoms", {}).get("urine", []),
                "notes": health_profile.get("symptoms", {}).get("notes"),
            },
        }
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the server running on localhost:8000?")
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ── Helpers ──
def format_response_with_citations(text: str, citations: list) -> str:
    """Replace inline [1], [2], … markers with hover-tooltip spans.

    Citations list is 0-indexed but numbered from 1.
    So citations[0] corresponds to [1] in the text, etc.
    """

    if not citations:
        return text

    # Build a lookup: citation number -> tooltip HTML
    tip_map: dict[int, str] = {}
    for idx, c in enumerate(citations):
        num = idx + 1
        title = html.escape(c.get("title") or "Unknown Paper")
        preview = html.escape((c.get("relevant_section") or "")[:250])
        tip_map[num] = (
            f'<span class="cite-ref">{num}'
            f'<span class="cite-tip">'
            f'<div class="cite-tip-title">{title}</div>'
            f'<div class="cite-tip-body">{preview}</div>'
            f'</span></span>'
        )

    # Replace every [N] in text with the tooltip span
    def _replace(m: re.Match) -> str:
        num = int(m.group(1))
        return tip_map.get(num, m.group(0))

    return re.sub(r"\[(\d+)\]", _replace, text)


def _ph_badge(ph: float) -> str:
    if 3.8 <= ph <= 4.5:
        status = "Normal Range"
    elif 4.5 < ph <= 5.0:
        status = "Borderline"
    else:
        status = "Elevated"
    return f'<span class="ph-badge">pH {ph} · {status}</span>'


# ── Page 1: Form ──
def show_form_page():
    st.markdown(
        '<div class="app-header"><h1>pHera</h1>'
        '<p>Evidence-based vaginal pH analysis</p></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Enter a pH reading and optional health context for evidence-based analysis."
    )

    with st.form(key="health_form"):
        with st.container(border=True):
            st.markdown("**pH Reading**")
            ph_value = st.number_input(
                "pH value", min_value=0.0, max_value=14.0,
                value=4.5, step=0.1,
            )

        st.divider()

        with st.container(border=True):
            st.markdown("**Demographics**")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input(
                    "Age", min_value=0, max_value=120,
                    value=None,
                )
            with col2:
                menstrual_cycle = st.selectbox(
                    "Menstrual status",
                    options=[
                        "Regular", "Irregular", "No period for 12+ months",
                        "Never had a period", "Perimenopausal", "Postmenopausal",
                    ],
                    index=None,
                )
            ethnic_backgrounds = st.multiselect(
                "Ethnic background",
                options=[
                    "African / Black", "North African", "Arab", "Middle Eastern",
                    "East Asian", "South Asian", "Southeast Asian",
                    "Central Asian / Caucasus",
                    "Latin American / Latina / Latinx / Hispanic",
                    "Sinti / Roma", "White / Caucasian / European",
                    "Mixed / Multiple ancestries", "Other",
                ],
            )

        st.divider()

        with st.container(border=True):
            st.markdown("**Medical History**")
            diagnoses = st.multiselect(
                "Diagnoses",
                options=[
                    "Adenomyosis", "Amenorrhea", "Bacterial vaginosis (BV)",
                    "Cushing's syndrome", "Diabetes", "Endometriosis",
                    "Intersex status", "Thyroid disorder", "Uterine fibroids",
                    "Polycystic ovary syndrome (PCOS)",
                    "Premature ovarian insufficiency (POI)", "Yeast infection",
                ],
            )

        st.divider()

        with st.container(border=True):
            st.markdown("**Birth Control**")
            col1, col2 = st.columns(2)
            with col1:
                bc_general = st.selectbox(
                    "General",
                    options=[
                        None, "No control", "Stopped in last 3 months",
                        "Emergency contraception in last 7 days",
                    ],
                    index=0,
                )
            with col2:
                bc_pill = st.selectbox(
                    "Pill",
                    options=[None, "Combined pill", "Progestin-only pill"],
                    index=0,
                )
            col3, col4 = st.columns(2)
            with col3:
                bc_iud = st.selectbox(
                    "IUD",
                    options=[None, "Hormonal IUD", "Copper IUD"],
                    index=0,
                )
            with col4:
                bc_other = st.multiselect(
                    "Other methods",
                    options=["Implant", "Injection", "Vaginal ring", "Patch"],
                )
            bc_permanent = st.multiselect(
                "Permanent",
                options=["Tubal ligation"],
            )

        st.divider()

        with st.container(border=True):
            st.markdown("**Hormone Therapy**")
            col1, col2 = st.columns(2)
            with col1:
                hormone_therapy = st.multiselect(
                    "Hormone therapy",
                    options=["Estrogen only", "Estrogen + Progestin"],
                )
            with col2:
                hrt = st.multiselect(
                    "HRT",
                    options=["Testosterone", "Estrogen blocker", "Puberty blocker"],
                )

        st.divider()

        with st.container(border=True):
            st.markdown("**Fertility Journey**")
            col1, col2 = st.columns(2)
            with col1:
                fertility_status = st.selectbox(
                    "Current status",
                    options=[
                        None, "Pregnant", "Had baby (last 12 months)",
                        "Not able to get pregnant", "Trying to conceive",
                    ],
                    index=0,
                )
            with col2:
                fertility_treatments = st.multiselect(
                    "Fertility treatments",
                    options=[
                        "Ovulation induction", "IUI", "IVF",
                        "Egg freezing", "Luteal progesterone",
                    ],
                )

        st.divider()

        with st.container(border=True):
            st.markdown("**Symptoms**")
            col1, col2 = st.columns(2)
            with col1:
                discharge = st.multiselect(
                    "Discharge type",
                    options=[
                        "No discharge", "Creamy", "Sticky", "Egg white",
                        "Clumpy white", "Grey and watery", "Yellow/Green", "Red/Brown",
                    ],
                )
            with col2:
                vulva_vagina = st.multiselect(
                    "Vulva/Vagina symptoms",
                    options=["None", "Dry", "Itchy", "Burning"],
                )
            col3, col4 = st.columns(2)
            with col3:
                smell = st.multiselect(
                    "Odor",
                    options=[
                        "None", 'Strong and unpleasant ("fishy")', "Sour",
                        "Chemical-like", "Very strong or rotten",
                    ],
                )
            with col4:
                urine = st.multiselect(
                    "Urinary symptoms",
                    options=["None", "Frequent urination", "Burning sensation"],
                )
            notes = st.text_area(
                "Additional notes",
                placeholder="Any other information (optional)",
                height=80,
            )

        submitted = st.form_submit_button("Analyze pH Reading", use_container_width=True)

    if submitted:
        # Build birth control info
        birth_control = None
        if bc_general or bc_pill or bc_iud or bc_other or bc_permanent:
            birth_control = {
                "general": bc_general,
                "pill": bc_pill,
                "iud": bc_iud,
                "other_methods": bc_other,
                "permanent": bc_permanent,
            }

        # Build fertility journey info
        fertility_journey = None
        if fertility_status or fertility_treatments:
            fertility_journey = {
                "current_status": fertility_status,
                "fertility_treatments": fertility_treatments,
            }

        health_profile = {
            "age": int(age) if age else None,
            "menstrual_cycle": menstrual_cycle,
            "diagnoses": diagnoses,
            "ethnic_backgrounds": ethnic_backgrounds,
            "birth_control": birth_control,
            "hormone_therapy": hormone_therapy,
            "hrt": hrt,
            "fertility_journey": fertility_journey,
            "symptoms": {
                "discharge": discharge,
                "vulva_vagina": vulva_vagina,
                "smell": smell,
                "urine": urine,
                "notes": notes,
            },
        }

        with st.status("Searching medical literature...", expanded=True) as status:
            st.write("Querying evidence database...")
            response = call_medical_rag_api(ph_value, health_profile)
            if response:
                status.update(label="Analysis complete", state="complete")
            else:
                status.update(label="Analysis failed", state="error")

        if response:
            st.session_state.ph_value = ph_value
            st.session_state.health_profile = health_profile
            st.session_state.session_id = response.get("session_id")
            st.session_state.first_response = response.get("agent_reply", "")
            st.session_state.first_citations = response.get("citations", [])
            st.session_state.disclaimers = response.get("disclaimers")
            st.session_state.processing_time_ms = response.get("processing_time_ms")
            st.session_state.chat_history = [
                {"role": "user", "content": f"pH: {ph_value}"},
                {
                    "role": "assistant",
                    "content": st.session_state.first_response,
                    "citations": st.session_state.first_citations,
                },
            ]
            st.session_state.page = "chat"
            st.rerun()


# ── Page 2: Chat ──
def _render_sidebar():
    """Render sidebar with pH badge, profile summary, and controls."""
    with st.sidebar:
        if st.session_state.ph_value is not None:
            st.markdown(_ph_badge(st.session_state.ph_value), unsafe_allow_html=True)
            st.divider()

        profile = st.session_state.health_profile
        if profile:
            st.markdown("**Profile Summary**")
            if profile.get("age"):
                st.caption(f"Age: {profile['age']}")
            if profile.get("ethnic_backgrounds"):
                st.caption(f"Ethnicity: {', '.join(profile['ethnic_backgrounds'])}")
            if profile.get("menstrual_cycle"):
                st.caption(f"Menstrual status: {profile['menstrual_cycle']}")
            if profile.get("diagnoses"):
                st.caption(f"Diagnoses: {', '.join(profile['diagnoses'])}")

            # Birth control
            bc = profile.get("birth_control")
            if bc:
                bc_items = []
                if bc.get("general"):
                    bc_items.append(bc["general"])
                if bc.get("pill"):
                    bc_items.append(bc["pill"])
                if bc.get("iud"):
                    bc_items.append(bc["iud"])
                bc_items.extend(bc.get("other_methods", []))
                bc_items.extend(bc.get("permanent", []))
                if bc_items:
                    st.caption(f"Birth control: {', '.join(bc_items)}")

            # Hormone therapy
            ht = profile.get("hormone_therapy", []) + profile.get("hrt", [])
            if ht:
                st.caption(f"Hormone therapy: {', '.join(ht)}")

            # Fertility journey
            fj = profile.get("fertility_journey")
            if fj:
                fj_items = []
                if fj.get("current_status"):
                    fj_items.append(fj["current_status"])
                fj_items.extend(fj.get("fertility_treatments", []))
                if fj_items:
                    st.caption(f"Fertility: {', '.join(fj_items)}")

            # Symptoms
            symptoms = profile.get("symptoms", {})
            active = []
            for key in ("discharge", "vulva_vagina", "smell", "urine"):
                vals = symptoms.get(key, [])
                if vals and vals != ["None"]:
                    active.extend(vals)
            if active:
                st.caption(f"Symptoms: {', '.join(active)}")
            st.divider()

        if st.button("New Analysis", use_container_width=True):
            for key in (
                "chat_history", "first_response", "first_citations",
                "session_id", "disclaimers", "processing_time_ms",
            ):
                st.session_state[key] = [] if key == "chat_history" else None
            st.session_state.page = "form"
            st.rerun()


def _render_message(content: str, citations: list, show_disclaimer: bool = False) -> None:
    """Render an assistant message with inline citation tooltips."""
    if citations:
        st.markdown(
            format_response_with_citations(content, citations),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(content)

    if show_disclaimer and st.session_state.disclaimers:
        disclaimers = st.session_state.disclaimers
        if isinstance(disclaimers, list):
            disclaimers = " ".join(disclaimers)
        st.caption(disclaimers)


def show_chat_page():
    _render_sidebar()

    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            citations = msg.get("citations", [])
            if msg["role"] == "assistant":
                # Show disclaimer only on the last assistant message
                is_last_assistant = (idx == len(st.session_state.chat_history) - 1)
                _render_message(content, citations, show_disclaimer=is_last_assistant)
            else:
                st.markdown(content)

    if follow_up := st.chat_input("Ask a follow-up question..."):
        st.session_state.chat_history.append({"role": "user", "content": follow_up})

        with st.chat_message("user"):
            st.markdown(follow_up)

        with st.chat_message("assistant"):
            with st.status("Searching medical literature...", expanded=True):
                response = call_medical_rag_api(
                    ph_value=st.session_state.ph_value,
                    health_profile=st.session_state.health_profile,
                    user_message=follow_up,
                    session_id=st.session_state.session_id,
                )

            if response:
                content = response.get("agent_reply", "")
                citations = response.get("citations", [])
                _render_message(content, citations, show_disclaimer=True)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": content,
                    "citations": citations,
                })
                if response.get("disclaimers"):
                    st.session_state.disclaimers = response["disclaimers"]

        if response:
            st.rerun()


def main():
    if st.session_state.page == "form":
        show_form_page()
    else:
        show_chat_page()


if __name__ == "__main__":
    main()
