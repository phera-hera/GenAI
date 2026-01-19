"""
pHera Medical Agent - Main Entry Point

Role-based interface:
- User Mode: For scientists/researchers testing the medical agent
- Admin Mode: For administrators managing papers and system
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="pHera Medical Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for user role
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

# Main title with professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    .role-card {
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background: #fafafa;
        margin-bottom: 1rem;
    }
    .role-title {
        font-size: 1.3rem;
        font-weight: 500;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .role-desc {
        color: #5a6c7d;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">pHera Medical Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Vaginal Health Analysis Platform</div>', unsafe_allow_html=True)
st.markdown("---")

# Role selection if not set
if st.session_state.user_role is None:
    st.markdown("### Select Your Role")
    st.markdown("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### User Mode")
        st.markdown("""
        **Test pH analysis only**

        Access to pH analysis testing for doctors, scientists, and researchers.
        """)

        if st.button("Continue as User", use_container_width=True, type="primary"):
            st.session_state.user_role = "user"
            st.rerun()

    with col2:
        st.markdown("#### Admin Mode")
        st.markdown("""
        **Full system access**

        Paper management, ingestion, system monitoring, and pH analysis testing.
        """)

        admin_password = st.text_input("Admin Password", type="password", key="admin_pass")

        if st.button("Continue as Admin", use_container_width=True):
            if admin_password == st.secrets.get("admin_password", "admin123"):
                st.session_state.user_role = "admin"
                st.rerun()
            else:
                st.error("Invalid admin password")

else:
    # Show current role and allow switching
    st.sidebar.markdown(f"**Current Role:** {st.session_state.user_role.title()}")
    
    # API Configuration in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Configuration")
    st.session_state.api_base_url = st.sidebar.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL of the Medical Agent API"
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Switch Role / Logout", use_container_width=True):
        st.session_state.user_role = None
        st.rerun()
    
    # Welcome message based on role
    if st.session_state.user_role == "user":
        st.markdown("""
        <div style="padding: 1rem; background: #e8f4f8; border-left: 4px solid #3498db; border-radius: 4px; margin-bottom: 1.5rem;">
            <strong>User Mode</strong><br/>
            <span style="color: #5a6c7d;">Navigate to <strong>Test pH Analysis</strong> in the sidebar to begin testing.</span>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.user_role == "admin":
        st.markdown("""
        <div style="padding: 1rem; background: #fef5e7; border-left: 4px solid #f39c12; border-radius: 4px; margin-bottom: 1.5rem;">
            <strong>Admin Mode</strong><br/>
            <span style="color: #5a6c7d;">Access all features via the sidebar.</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("pHera Medical Agent - Confidential Testing Environment")
