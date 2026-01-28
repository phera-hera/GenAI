"""
Medical RAG Streamlit Application

Professional dark-themed interface for medical pH analysis with research citations.
- Page 1: Form submission with health profile
- Page 2: Chat interface for follow-up questions
"""

import logging
import uuid
from typing import Any

import requests
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Medical pH Analysis",
    page_icon="⚕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark theme
st.markdown("""
    <style>
        /* Dark background */
        .stApp {
            background-color: #0a0e27;
            color: #e0e0e0;
        }

        /* Input fields - grey */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stMultiSelect > div > div > div {
            background-color: #2a2e3e !important;
            color: #e0e0e0 !important;
            border: 1px solid #444 !important;
        }

        /* Button - violet */
        .stButton > button {
            background-color: #7c3aed !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            width: 100%;
            padding: 10px !important;
        }

        .stButton > button:hover {
            background-color: #6d28d9 !important;
        }

        /* Text areas */
        .stTextArea > div > div > textarea {
            background-color: #2a2e3e !important;
            color: #e0e0e0 !important;
            border: 1px solid #444 !important;
        }

        /* Container styling */
        .stContainer {
            background-color: #0a0e27;
        }

        /* Citation box */
        .citation-box {
            background-color: #1a1f2e;
            border-left: 4px solid #7c3aed;
            padding: 14px;
            margin: 12px 0;
            border-radius: 4px;
            font-size: 0.95rem;
        }

        /* Chat message - user */
        .chat-user {
            background-color: #1a1f2e;
            padding: 14px;
            margin: 10px 0;
            border-radius: 6px;
            margin-left: 15%;
            font-size: 0.95rem;
        }

        /* Chat message - assistant */
        .chat-assistant {
            background-color: #2a2e3e;
            padding: 14px;
            margin: 10px 0;
            border-radius: 6px;
            margin-right: 15%;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        /* Title styling */
        h1 {
            color: #e0e0e0;
            text-align: center;
        }

        h3 {
            color: #b0b0b0;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== CONFIG =====
API_URL = "http://localhost:8000/api/v1/query"

# ===== SESSION STATE INITIALIZATION =====
if "page" not in st.session_state:
    st.session_state.page = "form"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "ph_value" not in st.session_state:
    st.session_state.ph_value = None

if "health_profile" not in st.session_state:
    st.session_state.health_profile = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "first_response" not in st.session_state:
    st.session_state.first_response = None

if "first_citations" not in st.session_state:
    st.session_state.first_citations = None


# ===== API FUNCTIONS =====
def call_medical_rag_api(
    ph_value: float,
    health_profile: dict[str, Any],
    user_message: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Call the medical RAG API endpoint.

    Args:
        ph_value: pH measurement value
        health_profile: User's health information
        user_message: User's actual question (for follow-ups)
        session_id: Session ID for conversation continuity

    Returns:
        Response dict or None on error
    """
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
            },
            "notes": health_profile.get("notes"),
        }

        logger.info(f"Calling API: {API_URL}")
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the server running on http://localhost:8000?")
        logger.error("API connection error")
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        logger.error("API timeout")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.exception(f"API error: {e}")
        return None


# ===== PAGE 1: FORM =====
def show_form_page():
    """Display the initial form for pH analysis."""
    st.title("Medical pH Analysis")
    st.markdown("---")
    st.markdown("#### Health Profile")
    st.markdown("Provide your health information for evidence-based analysis")
    st.markdown("")

    with st.form(key="health_form"):
        # pH Value (Required)
        st.markdown("**pH Reading** (Required)")
        ph_value = st.number_input(
            "Enter your vaginal pH value (0-14)",
            min_value=0.0,
            max_value=14.0,
            value=4.5,
            step=0.1,
            label_visibility="collapsed",
        )

        st.markdown("")

        # Age
        st.markdown("**Age**")
        age = st.number_input(
            "Your age",
            min_value=0,
            max_value=120,
            value=None,
            label_visibility="collapsed",
        )

        st.markdown("")

        # Menstrual Cycle
        st.markdown("**Menstrual Cycle Status**")
        menstrual_cycle = st.selectbox(
            "Select your menstrual status",
            options=[
                "Regular",
                "Irregular",
                "No period for 12+ months",
                "Never had period",
                "Perimenopause",
                "Postmenopause",
                "Not sure",
            ],
            index=None,
            label_visibility="collapsed",
        )

        st.markdown("")

        # Diagnoses
        st.markdown("**Medical History** (Select if applicable)")
        diagnoses = st.multiselect(
            "Select any relevant diagnoses",
            options=[
                "PCOS",
                "Endometriosis",
                "Thyroid disorder",
                "Diabetes",
                "BV",
                "Yeast infection",
                "None",
            ],
            label_visibility="collapsed",
        )

        st.markdown("")

        # Symptoms
        st.markdown("**Current Symptoms** (Select if applicable)")
        discharge = st.multiselect(
            "Discharge type",
            options=[
                "No discharge",
                "Creamy",
                "Sticky",
                "Egg white",
                "Clumpy white",
                "Grey and watery",
                "Yellow/Green",
                "Red/Brown",
            ],
            label_visibility="collapsed",
        )

        vulva_vagina = st.multiselect(
            "Vaginal/Vulva symptoms",
            options=["Dry", "Itchy", "Burning", "None"],
            label_visibility="collapsed",
        )

        smell = st.multiselect(
            "Odor",
            options=["None", "Fishy", "Sour", "Chemical-like", "Very strong/rotten"],
            label_visibility="collapsed",
        )

        urine = st.multiselect(
            "Urinary symptoms",
            options=["None", "Frequent urination", "Burning sensation"],
            label_visibility="collapsed",
        )

        st.markdown("")

        # Ethnic Background
        st.markdown("**Ethnic Background** (Optional)")
        ethnic_backgrounds = st.multiselect(
            "Select if applicable",
            options=[
                "South Asian",
                "East Asian",
                "African",
                "European",
                "Hispanic/Latino",
                "Middle Eastern",
                "Mixed",
                "Prefer not to say",
            ],
            label_visibility="collapsed",
        )

        st.markdown("")

        # Notes
        st.markdown("**Additional Notes** (Optional)")
        notes = st.text_area(
            "Any other information you'd like to share?",
            placeholder="Type here...",
            label_visibility="collapsed",
            height=80,
        )

        st.markdown("")

        # Submit Button
        submitted = st.form_submit_button(label="Analyze pH Reading", use_container_width=True)

    if submitted:
        # Build health profile
        health_profile = {
            "age": int(age) if age else None,
            "menstrual_cycle": menstrual_cycle,
            "diagnoses": [d for d in diagnoses if d != "None"],
            "symptoms": {
                "discharge": discharge,
                "vulva_vagina": vulva_vagina,
                "smell": smell,
                "urine": urine,
            },
            "ethnic_backgrounds": ethnic_backgrounds,
            "notes": notes,
        }

        # Call API (initial request - no user_message or session_id yet)
        with st.spinner("Analyzing your pH reading..."):
            response = call_medical_rag_api(ph_value, health_profile)

        if response:
            # Store in session state
            st.session_state.ph_value = ph_value
            st.session_state.health_profile = health_profile
            st.session_state.session_id = response.get("session_id")  # Store session ID from API
            st.session_state.first_response = response.get("agent_reply", "")
            st.session_state.first_citations = response.get("citations", [])

            # Initialize chat history with first exchange
            st.session_state.chat_history = [
                {"role": "user", "content": f"pH: {ph_value}"},
                {"role": "assistant", "content": st.session_state.first_response},
            ]

            # Switch to chat page
            st.session_state.page = "chat"
            st.rerun()


# ===== PAGE 2: CHAT =====
def show_chat_page():
    """Display the chat interface."""
    st.title("Medical pH Analysis")
    st.markdown("---")

    # Back to form button
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.page = "form"
            st.session_state.chat_history = []
            st.session_state.first_response = None
            st.session_state.first_citations = None
            st.session_state.session_id = None  # Reset session for new conversation
            st.rerun()

    st.markdown("")

    # Chat history
    with st.container():
        # Display chat messages
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant"><strong>Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

        # Display citations for last assistant message
        if st.session_state.first_citations and len(st.session_state.chat_history) > 1:
            if st.session_state.chat_history[-1]["role"] == "assistant":
                st.markdown("#### References")
                for i, citation in enumerate(st.session_state.first_citations, 1):
                    paper_id = citation.get('paper_id', 'Unknown')
                    title = citation.get('title', 'Unknown Paper')
                    preview = citation.get('relevant_section', '')
                    st.markdown(
                        f"""<div class="citation-box">
                        <strong>[{i}] {title}</strong><br>
                        {preview[:150]}...
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.markdown("")
    st.markdown("---")

    # Follow-up input
    st.markdown("#### Follow-up Question")
    follow_up = st.text_input(
        "Your question:",
        placeholder="Ask a follow-up question about your results...",
        label_visibility="collapsed",
    )

    if st.button("Send", use_container_width=True):
        if follow_up.strip():
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": follow_up})

            # Call API with follow-up question and session_id for context
            with st.spinner("Analyzing your question..."):
                response = call_medical_rag_api(
                    ph_value=st.session_state.ph_value,
                    health_profile=st.session_state.health_profile,
                    user_message=follow_up,  # Send actual follow-up question
                    session_id=st.session_state.session_id,  # Reuse session for memory
                )

            if response:
                # Add assistant response
                assistant_response = response.get("agent_reply", "")
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                st.session_state.first_citations = response.get("citations", [])
                st.rerun()
        else:
            st.warning("Please enter a question.")


def main():
    """Route to appropriate page based on session state."""
    if st.session_state.page == "form":
        show_form_page()
    else:
        show_chat_page()


if __name__ == "__main__":
    main()

