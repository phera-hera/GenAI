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

# Main title
st.title("pHera Medical Agent")
st.markdown("### Vaginal Health Analysis Platform")
st.markdown("---")

# Role selection if not set
if st.session_state.user_role is None:
    st.header("Welcome! Please Select Your Role")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### User Mode")
        st.markdown("""
        **For Scientists & Researchers**
        
        Test the medical agent with:
        - pH analysis testing
        - Health profile simulations
        - Result validation
        - API health monitoring
        
        *No technical knowledge required*
        """)
        
        if st.button("Continue as User", use_container_width=True, type="primary"):
            st.session_state.user_role = "user"
            st.rerun()
    
    with col2:
        st.markdown("### Admin Mode")
        st.markdown("""
        **For System Administrators**
        
        Manage the system:
        - Paper management (add/delete)
        - Database operations
        - System monitoring
        - Advanced configurations
        
        *Technical access required*
        """)
        
        admin_password = st.text_input("Admin Password", type="password", key="admin_pass")
        
        if st.button("Continue as Admin", use_container_width=True):
            # Simple password check (in production, use proper authentication)
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
        st.success("Welcome! You are in User Mode")
        st.markdown("""
        ### Getting Started
        
        Use the sidebar to navigate to:
        - **Test pH Analysis**: Test the medical agent with various scenarios
        
        The interface is designed to be simple and straightforward - no technical knowledge needed!
        """)
        
        st.info("💡 **Tip**: Go to 'Test pH Analysis' to start testing different health scenarios.")
        
    elif st.session_state.user_role == "admin":
        st.success("Welcome! You are in Admin Mode")
        st.markdown("""
        ### Admin Dashboard
        
        Use the sidebar to navigate to:
        - **Test pH Analysis**: Test the medical agent (same as user mode)
        - **Paper Management**: View and delete research papers
        - **API Health**: Monitor API status and diagnostics
        
        Admin access provides paper management and system monitoring capabilities.
        """)
        
        st.warning("⚠️ **Admin Notice**: Paper deletion operations are permanent. Always verify before deleting.")

# Footer
st.markdown("---")
st.caption("pHera Medical Agent - Confidential Testing Environment")
