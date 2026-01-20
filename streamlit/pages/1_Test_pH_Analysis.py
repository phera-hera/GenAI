"""
pH Analysis Testing Page

Available for both User and Admin roles.
Simple interface for testing the medical agent with various health scenarios.
"""

import streamlit as st
import requests
from typing import Dict, Any

# Page config
st.set_page_config(
    page_title="Test pH Analysis",
    page_icon="🔬",
    layout="wide",
)

# Check if user is logged in
if "user_role" not in st.session_state or st.session_state.user_role is None:
    st.warning("Please select your role from the Home page first.")
    st.stop()

# Get API URL from session
API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

# Professional styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
    }
    
    /* Header Styles */
    h1, h2, h3 {
        color: #4caf50 !important; /* Standard Green */
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    h4 {
        color: #81c784 !important; /* Soft Green */
    }
    
    /* Metric Card */
    .metric-card {
        background: #121212;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #2e7d32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    
    .section-header {
        color: #4caf50;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Button */
    div.stButton > button:first-child {
        background-color: #0a2f10;
        color: #a5d6a7;
        border: 1px solid #1b5e20;
    }
    div.stButton > button:first-child:hover {
        background-color: #1b5e20;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.3);
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Hide sidebar for non-admin users
if st.session_state.user_role != "admin":
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add a Home button for users since they have no sidebar
    if st.button("← Back to Home"):
        st.session_state.user_role = None
        st.switch_page("Home.py")

st.title("pH Analysis Testing")
st.markdown("---")

# Medical Disclaimer
with st.expander("Medical Disclaimer"):
    st.info("""
    This system provides informational analysis only and does not diagnose conditions or replace professional medical advice. Always consult a qualified healthcare provider for medical concerns.
    """)

# Main content - Single Column Layout

st.markdown("#### Required Information")

# pH Value (Required)
ph_value = st.number_input(
    "pH Value*",
    min_value=0.0,
    max_value=14.0,
    value=4.5,
    step=0.1
)

st.markdown("---")
st.markdown("#### Basic Information")

# Age
age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=None,
    placeholder="Enter age..."
)

# Ethnic Background
ethnic_backgrounds = st.multiselect(
    "Ethnic Background(s)",
    options=[
        "African / Black",
        "North African",
        "Arab",
        "Middle Eastern",
        "East Asian",
        "South Asian",
        "Southeast Asian",
        "Central Asian / Caucasus",
        "Latin American / Latina / Latinx / Hispanic",
        "Sinti / Roma",
        "White / Caucasian / European",
        "Mixed / Multiple ancestry",
        "Other"
    ]
)

# Diagnoses
diagnoses = st.multiselect(
    "Hormone-Related Diagnoses",
    options=[
        "Adenomyosis",
        "Amenorrhea",
        "Cushing's syndrome",
        "Diabetes",
        "Endometriosis",
        "Intersex status",
        "Thyroid disorder",
        "Uterine fibroids",
        "Polycystic ovary syndrome (PCOS)",
        "Premature ovarian insufficiency (POI)"
    ]
)

st.markdown("---")
st.markdown("#### Hormone Status")

# Menstrual Cycle
menstrual_cycle = st.selectbox(
    "Menstrual Cycle Status",
    options=[
        "Regular",
        "Irregular",
        "No period for 12+ months",
        "Never had a period",
        "Perimenopause",
        "Postmenopause"
    ],
    index=0
)

# Birth Control
with st.expander("Birth Control"):
    bc_general = st.selectbox(
        "General Status",
        options=[
            "None",
            "No birth control or hormonal birth control",
            "Stopped birth control in the last 3 months",
            "Morning after-pill / emergency contraception in the last 7 days"
        ],
        index=0
    )

    bc_pill = st.selectbox(
        "Pill Type",
        options=["None", "Combined pill", "Progestin-only pill"],
        index=0
    )

    bc_iud = st.selectbox(
        "IUD Type",
        options=["None", "Hormonal IUD", "Copper IUD"],
        index=0
    )

    bc_other = st.multiselect(
        "Other Hormonal Methods",
        options=[
            "Contraceptive implant",
            "Contraceptive injection",
            "Vaginal ring",
            "Patch"
        ]
    )

    bc_permanent = st.multiselect(
        "Permanent Methods",
        options=["Tubal ligation"]
    )

# Hormone Therapy
hormone_therapy = st.multiselect(
    "Hormone Therapy",
    options=["Estrogen only", "Estrogen + progestin"]
)

hrt = st.multiselect(
    "HRT",
    options=["Testosterone", "Estrogen blocker", "Puberty blocker"]
)

st.markdown("---")
st.markdown("#### Fertility Journey")

fertility_status = st.selectbox(
    "Current Status",
    options=[
        "None",
        "I am pregnant",
        "I had a baby (last 12 months)",
        "I am not able to get pregnant",
        "I am trying to conceive"
    ],
    index=0
)

fertility_treatments = st.multiselect(
    "Treatments (last 3 months)",
    options=[
        "Ovulation induction",
        "Intrauterine insemination (IUI)",
        "In vitro fertilisation (IVF)",
        "Egg freezing stimulation",
        "Luteal progesterone"
    ]
)

st.markdown("---")
st.markdown("#### Current Symptoms")

# Discharge
discharge_symptoms = st.multiselect(
    "Discharge",
    options=[
        "No discharge",
        "Creamy",
        "Sticky",
        "Egg white",
        "Clumpy white",
        "Grey and watery",
        "Yellow / Green",
        "Red / Brown"
    ]
)

# Vulva & Vagina
vulva_symptoms = st.multiselect(
    "Vulva & Vagina",
    options=["Dry", "Itchy"]
)

# Smell
smell_symptoms = st.multiselect(
    "Smell",
    options=[
        "Strong and unpleasant (fishy)",
        "Sour",
        "Chemical-like",
        "Very strong or rotten"
    ]
)

# Urine
urine_symptoms = st.multiselect(
    "Urine",
    options=["Frequent urination", "Burning sensation"]
)

st.markdown("---")
st.markdown("#### Additional Notes")

# Notes
notes = st.text_area(
    "Notes",
    placeholder="Optional: Add any extra symptoms or context...",
    label_visibility="collapsed"
)

st.markdown("---")

# Submit button
submit_button = st.button("Analyze pH", use_container_width=True, type="primary")

if submit_button:
    # Build the request payload
    request_data: Dict[str, Any] = {
        "ph_value": ph_value,
        "age": age if age is not None and age > 0 else None,
    }
    
    # Add optional fields
    if diagnoses:
        request_data["diagnoses"] = diagnoses
    
    if ethnic_backgrounds:
        request_data["ethnic_backgrounds"] = ethnic_backgrounds
    
    if menstrual_cycle:
        request_data["menstrual_cycle"] = menstrual_cycle
    
    # Birth control
    birth_control_data = {}
    if bc_general and bc_general != "None":
        birth_control_data["general"] = bc_general
    if bc_pill and bc_pill != "None":
        birth_control_data["pill"] = bc_pill
    if bc_iud and bc_iud != "None":
        birth_control_data["iud"] = bc_iud
    if bc_other:
        birth_control_data["other_methods"] = bc_other
    if bc_permanent:
        birth_control_data["permanent"] = bc_permanent
    
    if birth_control_data:
        request_data["birth_control"] = birth_control_data
    
    if hormone_therapy:
        request_data["hormone_therapy"] = hormone_therapy
    
    if hrt:
        request_data["hrt"] = hrt
    
    # Fertility journey
    fertility_data = {}
    if fertility_status and fertility_status != "None":
        fertility_data["current_status"] = fertility_status
    if fertility_treatments:
        fertility_data["fertility_treatments"] = fertility_treatments
    
    if fertility_data:
        request_data["fertility_journey"] = fertility_data
    
    # Symptoms
    symptoms_data = {}
    if discharge_symptoms:
        symptoms_data["discharge"] = discharge_symptoms
    if vulva_symptoms:
        symptoms_data["vulva_vagina"] = vulva_symptoms
    if smell_symptoms:
        symptoms_data["smell"] = smell_symptoms
    if urine_symptoms:
        symptoms_data["urine"] = urine_symptoms
    
    if symptoms_data:
        request_data["symptoms"] = symptoms_data
    
    if notes:
        request_data["notes"] = notes
    
    # Show request payload for admin users
    if st.session_state.user_role == "admin":
        with st.expander("Request Payload (Admin Debug)"):
            st.json(request_data)
    
    # Make API call
    try:
        with st.spinner("Analyzing pH and consulting medical research..."):
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("Analysis Complete")
                
                # Display results
                st.markdown("---")
                st.header("Analysis Results")
                
                # Risk Level with color coding
                risk_level = result.get("risk_level", "UNKNOWN")
                risk_colors = {
                    "NORMAL": "#76FF03",  # Bright Green
                    "MONITOR": "#FFEA00", # Bright Yellow
                    "CONCERNING": "#FF9100", # Bright Orange
                    "URGENT": "#FF1744"   # Bright Red
                }
                risk_color = risk_colors.get(risk_level, "#B0BEC5")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("pH Value", result.get("ph_value"))
                with col2:
                    st.markdown(f"**Risk Level**")
                    st.markdown(f"<span style='color:{risk_color}; font-size:24px; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
                with col3:
                    st.metric("Processing Time", f"{result.get('processing_time_ms', 0)} ms")
                
                # Summary
                st.subheader("Summary")
                st.info(result.get("summary", "No summary available"))
                
                # Main Content
                st.subheader("Detailed Analysis")
                st.markdown(result.get("main_content", "No detailed content available"))
                
                # Personalized Insights
                if result.get("personalized_insights"):
                    st.subheader("Personalized Insights")
                    for insight in result["personalized_insights"]:
                        st.markdown(f"- {insight}")
                
                # Next Steps
                if result.get("next_steps"):
                    st.subheader("Recommended Next Steps")
                    for step in result["next_steps"]:
                        st.markdown(f"- {step}")
                
                # Citations
                if result.get("citations"):
                    st.subheader("Research Citations")
                    for i, citation in enumerate(result["citations"], 1):
                        with st.expander(f"Citation {i}: {citation.get('title', 'Untitled')}"):
                            if citation.get("authors"):
                                st.markdown(f"**Authors:** {citation['authors']}")
                            if citation.get("doi"):
                                st.markdown(f"**DOI:** {citation['doi']}")
                            if citation.get("relevant_section"):
                                st.markdown(f"**Relevant Section:**")
                                st.text(citation['relevant_section'])
                
                # Disclaimers
                st.markdown("---")
                st.warning(result.get("disclaimers", "This is not medical advice."))
                
                # Raw response for admin
                if st.session_state.user_role == "admin":
                    with st.expander("Full API Response (Admin Debug)"):
                        st.json(result)
            
            else:
                st.error(f"API Error: {response.status_code}")
                if st.session_state.user_role == "admin":
                    st.json(response.json())
                else:
                    st.info("Please contact an administrator if this issue persists.")
    
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the API. Make sure the server is running.")
    except requests.exceptions.Timeout:
        st.error("Timeout Error: The request took too long. Please try again.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
