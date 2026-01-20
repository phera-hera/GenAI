"""
pHera Medical Agent - Main Entry Point

Role-based interface:
- User Mode: For scientists/researchers testing the medical agent
- Admin Mode: For administrators managing papers and system
"""

import streamlit as st
import time

# Page config
st.set_page_config(
    page_title="pHera Medical Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state for user role
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

# Styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #050505; /* Ultra Dark background */
        color: #e0e0e0;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
    }

    /* Body text styling */
    body, p, div, span, label {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
    }

    /* Header Styles - pHera Logo Text */
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
        font-size: 6rem;
        font-weight: 800;
        background: linear-gradient(to right, #1A4D2E, #0F5257);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-align: center;
        margin-top: 2rem;
        letter-spacing: -3px;
    }

    /* Sub-header / Tagline */
    .sub-header {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
        font-size: 1.8rem;
        color: #3D8B6F; /* Muted Sage */
        font-weight: 500;
        margin-bottom: 3rem;
        text-align: center;
        line-height: 1.4;
        font-style: italic;
    }

    /* Custom Button Styling - Primary Action */
    div.stButton > button:first-child {
        background-color: #1A4D2E; /* Forest Green */
        color: #E8F5E9; /* Very Light Sage */
        border: 1px solid #245953;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', system-ui, sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    div.stButton > button:first-child:hover {
        background-color: #0F5257; /* Dark Teal */
        box-shadow: 0 0 20px rgba(15, 82, 87, 0.6);
        border-color: #245953;
        color: white;
    }

    /* Hide sidebar by default if user is not admin */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        color: #0F5257 !important;
    }
</style>
""", unsafe_allow_html=True)

# Admin logic - Show sidebar only for admin
if st.session_state.user_role == "admin":
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: block !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content for Admin
    st.sidebar.markdown(f"**Current Role:** {st.session_state.user_role.title()}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Configuration")
    st.session_state.api_base_url = st.sidebar.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL of the Medical Agent API"
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.user_role = None
        st.rerun()

# Logic for logged in User (Redirect to testing)
if st.session_state.user_role == "user":
    st.switch_page("pages/1_Test_pH_Analysis.py")

# Main Landing Page Content (When not logged in)
if st.session_state.user_role is None:
    # Centered Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo Text
        st.markdown('<div class="main-header">pHera</div>', unsafe_allow_html=True)
        
        # New Tagline
        st.markdown('<div class="sub-header">The first ultra-precise and personalised<br>vaginal health test</div>', unsafe_allow_html=True)
        
        # Start Button
        if st.button("Start pH Analysis", type="primary", use_container_width=True):
            st.session_state.user_role = "user"
            st.rerun()
            
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        # Discreet Admin Access
        with st.expander("Admin Access"):
            admin_password = st.text_input("Password", type="password", key="admin_pass")
            if st.button("Login as Admin"):
                if admin_password == st.secrets.get("admin_password", "admin123"):
                    st.session_state.user_role = "admin"
                    st.rerun()
                else:
                    st.error("Invalid password")

elif st.session_state.user_role == "admin":
    # Admin Dashboard Landing
    st.markdown("## Admin Dashboard")
    st.info("Welcome Admin. Use the sidebar to manage papers, monitor API health, or test the analysis.")
