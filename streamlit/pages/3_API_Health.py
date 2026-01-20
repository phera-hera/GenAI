"""
API Health Check Page

Available for both User and Admin roles.
Monitor API status and configuration.
"""

import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="API Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check if user is logged in
if "user_role" not in st.session_state or st.session_state.user_role is None:
    st.warning("Please select your role from the Home page first.")
    st.stop()

# Admin only page
if st.session_state.user_role != "admin":
    st.error("Access Denied: API Health monitoring is only available to administrators.")
    st.info("Regular users don't need to worry about API health - if there's an issue, you'll see a clear error message when testing.")
    st.stop()

# Get API URL from session
API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

# Professional styling
st.markdown("""
<style>
    /* Global Styles */
    h1, h2, h3 {
        color: #8BC34A !important;
    }
    
    .status-good {
        color: #76FF03;
        font-weight: bold;
    }
    .status-bad {
        color: #FF1744;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("API Health Check")
st.markdown("---")

# Quick status check
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**API Base URL:** `{API_BASE_URL}`")
with col2:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

st.markdown("---")

# Two columns for health checks
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Health Check")

    if st.button("Check API Health", use_container_width=True, type="primary"):
        try:
            with st.spinner("Checking..."):
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)

                if response.status_code == 200:
                    result = response.json()
                    st.success("✓ API is healthy")

                    with st.expander("Health Details"):
                        st.json(result)
                else:
                    st.error(f"✗ Status: {response.status_code}")
                    with st.expander("Error Details"):
                        st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("✗ Connection failed - API server not running")
        except requests.exceptions.Timeout:
            st.error("✗ Request timed out")
        except Exception as e:
            st.error(f"✗ Error: {str(e)}")

with col2:
    st.markdown("#### API Information")

    if st.button("Get API Info", use_container_width=True, type="primary"):
        try:
            with st.spinner("Fetching..."):
                response = requests.get(f"{API_BASE_URL}/", timeout=5)

                if response.status_code == 200:
                    result = response.json()
                    st.success("✓ Retrieved successfully")

                    if "name" in result:
                        st.markdown(f"**Name:** {result['name']}")
                    if "version" in result:
                        st.markdown(f"**Version:** {result['version']}")
                    if "description" in result:
                        st.markdown(f"**Description:** {result['description']}")

                    with st.expander("Full Information"):
                        st.json(result)
                else:
                    st.error(f"✗ Status: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("✗ Connection failed")
        except requests.exceptions.Timeout:
            st.error("✗ Request timed out")
        except Exception as e:
            st.error(f"✗ Error: {str(e)}")

st.markdown("---")

# Advanced diagnostics
st.markdown("---")
st.markdown("#### Advanced Diagnostics")

with st.expander("Test Endpoints"):
    endpoint = st.text_input("Custom Endpoint", placeholder="/api/v1/...")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("GET Request"):
            if endpoint:
                try:
                    response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
                    st.write(f"**Status:** {response.status_code}")
                    st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("Test /docs"):
            try:
                response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
                if response.status_code == 200:
                    st.success(f"✓ Accessible at: {API_BASE_URL}/docs")
                else:
                    st.warning("Documentation not enabled")
            except Exception as e:
                st.error(f"Error: {e}")

    with col3:
        if st.button("Test /openapi.json"):
            try:
                response = requests.get(f"{API_BASE_URL}/openapi.json", timeout=5)
                if response.status_code == 200:
                    st.success("✓ OpenAPI available")
                    with st.expander("Schema"):
                        st.json(response.json())
                else:
                    st.warning("Schema not enabled")
            except Exception as e:
                st.error(f"Error: {e}")

# Common issues reference
with st.expander("Common Issues"):
    st.markdown("""
    **Connection Refused**: API server not running
    **503 Unavailable**: Azure OpenAI not configured (check .env)
    **Timeout**: Server overloaded or complex query
    **404 Not Found**: Endpoint doesn't exist
    """)
