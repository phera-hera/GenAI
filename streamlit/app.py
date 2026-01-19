"""
Streamlit Frontend for pHera Medical Agent Testing

A simple testing interface for the medical RAG agent API.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any

# Page config
st.set_page_config(
    page_title="pHera Medical Agent - Test Interface",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Main title
st.title("pHera Medical Agent - Test Interface")
st.markdown("---")

# API Configuration at top
col1, col2 = st.columns([3, 1])
with col1:
    API_BASE_URL = st.text_input(
        "API Base URL", 
        value="http://localhost:8000",
        help="Base URL of the Medical Agent API"
    )
with col2:
    st.write("")  # Spacer
    st.write("")  # Spacer

st.markdown("---")

# Medical Disclaimer
with st.expander("Medical Disclaimer - Please Read"):
    st.warning("""
    **IMPORTANT NOTICE:**
    
    This system is **purely informational** and is **NOT** intended to:
    - Diagnose medical conditions
    - Prescribe treatments or medications
    - Replace professional medical advice
    
    Always consult with a qualified healthcare provider for medical concerns.
    """)

# Create tabs for better organization
tab1, tab2 = st.tabs(["pH Analysis", "API Health Check"])

with tab1:
    st.header("pH Analysis Request")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Required Information")
        
        # pH Value (Required)
        ph_value = st.number_input(
            "pH Value*",
            min_value=0.0,
            max_value=14.0,
            value=4.5,
            step=0.1,
            help="Enter the pH value from test strip (0-14)"
        )
        
        st.subheader("Basic Information")
        
        # Age
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=25,
            help="User's age"
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
            ],
            help="Select one or more ethnic backgrounds"
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
            ],
            help="Select any applicable diagnoses"
        )
        
        st.subheader("Hormone Status")
        
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
        with st.expander("Birth Control Information"):
            bc_general = st.selectbox(
                "General Birth Control Status",
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
            "Hormone Replacement Therapy (HRT)",
            options=["Testosterone", "Estrogen blocker", "Puberty blocker"]
        )
    
    with col2:
        st.subheader("Fertility Journey")
        
        fertility_status = st.selectbox(
            "Current Fertility Status",
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
            "Fertility Treatments (last 3 months)",
            options=[
                "Ovulation induction",
                "Intrauterine insemination (IUI)",
                "In vitro fertilisation (IVF)",
                "Egg freezing stimulation",
                "Luteal progesterone"
            ]
        )
        
        st.subheader("Current Symptoms")
        
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
            ],
            help="Discharge varies by cycle, hygiene, medications, and stress"
        )
        
        # Vulva & Vagina
        vulva_symptoms = st.multiselect(
            "Vulva & Vagina",
            options=["Dry", "Itchy"],
            help="Occasional dryness or itchiness is normal after shaving or new products"
        )
        
        # Smell
        smell_symptoms = st.multiselect(
            "Smell",
            options=[
                "Strong and unpleasant (fishy)",
                "Sour",
                "Chemical-like",
                "Very strong or rotten"
            ],
            help="A healthy vagina can have natural metallic, musky, earthy, or tangy scents"
        )
        
        # Urine
        urine_symptoms = st.multiselect(
            "Urine",
            options=["Frequent urination", "Burning sensation"],
            help="Normal to urinate more after fluids, coffee, or during stress"
        )
        
        # Notes
        notes = st.text_area(
            "Additional Notes",
            placeholder="Add any extra symptoms or how you've been feeling...",
            help="These notes are for app context only and NOT sent to the medical agent"
        )
    
    st.markdown("---")
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button("Analyze pH", use_container_width=True, type="primary")
    
    if submit_button:
        # Build the request payload
        request_data: Dict[str, Any] = {
            "ph_value": ph_value,
            "age": age if age > 0 else None,
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
        
        # Show request payload
        with st.expander("Request Payload (Debug)"):
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
                        "NORMAL": "#28a745",
                        "MONITOR": "#ffc107",
                        "CONCERNING": "#fd7e14",
                        "URGENT": "#dc3545"
                    }
                    risk_color = risk_colors.get(risk_level, "#6c757d")
                    
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
                    
                    # Raw response
                    with st.expander("Full API Response (Debug)"):
                        st.json(result)
                
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())
        
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API. Make sure the server is running.")
        except requests.exceptions.Timeout:
            st.error("Timeout Error: The request took too long. Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    st.header("API Health Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Check API Health", use_container_width=True):
            try:
                with st.spinner("Checking API health..."):
                    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("API is healthy")
                        st.json(result)
                    else:
                        st.error(f"API returned status code: {response.status_code}")
                        st.json(response.json())
            
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("Get API Info", use_container_width=True):
            try:
                with st.spinner("Fetching API info..."):
                    response = requests.get(f"{API_BASE_URL}/", timeout=5)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("API Info retrieved")
                        st.json(result)
                    else:
                        st.error(f"API returned status code: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer with quick tips
st.markdown("---")
st.markdown("### Quick Tips")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Getting Started**
    - Start the API server first
    - Default URL: `http://localhost:8000`
    - Only pH value is required
    """)

with col2:
    st.markdown("""
    **Testing**
    - Try different pH values (3.8-7.0)
    - Test with various symptom combinations
    - Check different risk levels
    """)

with col3:
    st.markdown("""
    **Notes**
    - User notes are for context only
    - Notes are NOT analyzed by agent
    - All other fields are optional
    """)

st.markdown("---")
st.caption("Built for pHera Medical Agent Testing")
