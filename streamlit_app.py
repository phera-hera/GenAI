"""Medical RAG Streamlit Application"""

import html
import logging
import uuid
from typing import Any

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="pHera · pH Analysis",
    page_icon="⚕",
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
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .ph-normal { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .ph-borderline { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .ph-abnormal { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

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

    /* ── Citations ── */
    .citation {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #818cf8;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.7em;
        vertical-align: super;
        line-height: 1;
        margin: 0 1px;
        padding: 1px 5px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    .citation:hover {
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.2);
    }

    .citation .tooltip {
        visibility: hidden;
        width: min(380px, 85vw);
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
        transition: opacity 0.2s ease, transform 0.2s ease;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        font-size: 0.85rem;
        line-height: 1.5;
        overflow: hidden;
        pointer-events: none;
    }
    .citation:hover .tooltip {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) scale(1);
        pointer-events: auto;
    }

    .tooltip-header {
        padding: 10px 14px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(99, 102, 241, 0.04));
        border-bottom: 1px solid rgba(99, 102, 241, 0.12);
        font-weight: 600;
        color: #e2e8f0;
        font-size: 0.84rem;
    }
    .tooltip-body {
        padding: 10px 14px;
        color: #94a3b8;
        font-size: 0.8rem;
        line-height: 1.6;
        display: -webkit-box;
        -webkit-line-clamp: 5;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .citation .tooltip::after {
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
    if not citations:
        return text
    markers = ""
    for i, c in enumerate(citations, 1):
        title = html.escape(c.get("title") or "Unknown Paper")
        preview = html.escape((c.get("relevant_section") or "")[:300])
        markers += (
            f'<span class="citation">{i}'
            f'<span class="tooltip">'
            f'<div class="tooltip-header">{title}</div>'
            f'<div class="tooltip-body">{preview}</div>'
            f'</span></span>'
        )
    return f"{text} {markers}"


def render_reference_list(citations: list) -> None:
    """Render an expandable reference list below the response."""
    if not citations:
        return
    with st.expander("References"):
        for i, c in enumerate(citations, 1):
            title = c.get("title") or "Unknown Paper"
            author = c.get("authors") or "Unknown Author"
            doi = c.get("doi")
            preview = (c.get("relevant_section") or "")[:300]

            if doi:
                doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
                st.markdown(f"**[{i}]** {title} — {author} ([DOI]({doi_url}))")
            else:
                st.markdown(f"**[{i}]** {title} — {author}")

            if preview:
                st.caption(f'"{preview}..."')


def _ph_badge(ph: float) -> str:
    if 3.8 <= ph <= 4.5:
        cls, label = "ph-normal", "Normal"
    elif 4.5 < ph <= 5.0:
        cls, label = "ph-borderline", "Borderline"
    else:
        cls, label = "ph-abnormal", "Abnormal"
    return f'<span class="ph-badge {cls}">pH {ph} · {label}</span>'


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
                    "Adenomyosis", "Amenorrhea", "Cushing's syndrome", "Diabetes",
                    "Endometriosis", "Intersex status", "Thyroid disorder",
                    "Uterine fibroids", "Polycystic ovary syndrome (PCOS)",
                    "Premature ovarian insufficiency (POI)",
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
                    options=["Dry", "Itchy", "Burning", "None"],
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
        health_profile = {
            "age": int(age) if age else None,
            "menstrual_cycle": menstrual_cycle,
            "diagnoses": diagnoses,
            "symptoms": {
                "discharge": discharge,
                "vulva_vagina": vulva_vagina,
                "smell": smell,
                "urine": urine,
                "notes": notes,
            },
            "ethnic_backgrounds": ethnic_backgrounds,
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
            if profile.get("menstrual_cycle"):
                st.caption(f"Menstrual status: {profile['menstrual_cycle']}")
            if profile.get("diagnoses"):
                st.caption(f"Diagnoses: {', '.join(profile['diagnoses'])}")
            symptoms = profile.get("symptoms", {})
            active = []
            for key in ("discharge", "vulva_vagina", "smell", "urine"):
                vals = symptoms.get(key, [])
                if vals and vals != ["None"]:
                    active.extend(vals)
            if active:
                st.caption(f"Symptoms: {', '.join(active)}")
            st.divider()

        if st.session_state.processing_time_ms is not None:
            st.caption(f"Processing time: {st.session_state.processing_time_ms}ms")

        if st.button("New Analysis", use_container_width=True):
            for key in (
                "chat_history", "first_response", "first_citations",
                "session_id", "disclaimers", "processing_time_ms",
            ):
                st.session_state[key] = [] if key == "chat_history" else None
            st.session_state.page = "form"
            st.rerun()


def _render_message(content: str, citations: list) -> None:
    """Render an assistant message with citations and reference list."""
    if citations:
        st.markdown(
            format_response_with_citations(content, citations),
            unsafe_allow_html=True,
        )
        render_reference_list(citations)
    else:
        st.markdown(content)


def show_chat_page():
    _render_sidebar()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            citations = msg.get("citations", [])
            if msg["role"] == "assistant":
                _render_message(content, citations)
            else:
                st.markdown(content)

    # Copy last response button
    last_assistant = None
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "assistant":
            last_assistant = msg["content"]
            break
    if last_assistant:
        if st.button("Copy response to clipboard", key="copy_btn"):
            escaped = last_assistant.replace("`", "\\`")
            st.components.v1.html(
                f"<script>navigator.clipboard.writeText(`{escaped}`)</script>",
                height=0,
            )
            st.toast("Copied to clipboard")

    # Processing time
    if st.session_state.processing_time_ms is not None:
        st.caption(f"Analysis completed in {st.session_state.processing_time_ms}ms")

    # Disclaimer
    if st.session_state.disclaimers:
        disclaimers = st.session_state.disclaimers
        if isinstance(disclaimers, list):
            disclaimers = " ".join(disclaimers)
        st.caption(disclaimers)

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
                _render_message(content, citations)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": content,
                    "citations": citations,
                })
                st.session_state.processing_time_ms = response.get(
                    "processing_time_ms", st.session_state.processing_time_ms
                )
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
