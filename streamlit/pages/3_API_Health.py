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
)

# Check if user is logged in
if "user_role" not in st.session_state or st.session_state.user_role is None:
    st.warning("Please select your role from the Home page first.")
    st.stop()

# Get API URL from session
API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

# Main title
st.title("API Health Check")
st.markdown("Monitor the status and configuration of the Medical Agent API")
st.markdown("---")

# Quick status check
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Refresh Status", use_container_width=True):
        st.rerun()

st.markdown("### Current Configuration")
st.info(f"**API Base URL:** `{API_BASE_URL}`")

st.markdown("---")

# Two columns for health checks
col1, col2 = st.columns(2)

with col1:
    st.subheader("Health Check")
    st.markdown("Verify the API is running and accessible")
    
    if st.button("Check API Health", use_container_width=True, type="primary"):
        try:
            with st.spinner("Checking API health..."):
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✓ API is healthy and running")
                    
                    # Display health details
                    with st.expander("Health Details"):
                        st.json(result)
                else:
                    st.error(f"✗ API returned status code: {response.status_code}")
                    if st.session_state.user_role == "admin":
                        with st.expander("Error Details"):
                            st.json(response.json())
        
        except requests.exceptions.ConnectionError:
            st.error("✗ Could not connect to the API")
            st.warning("**Troubleshooting:**")
            st.markdown("""
            - Verify the API server is running
            - Check the API Base URL in the sidebar
            - Ensure no firewall is blocking the connection
            """)
        except requests.exceptions.Timeout:
            st.error("✗ Request timed out")
            st.info("The API is taking too long to respond. It might be under heavy load.")
        except Exception as e:
            st.error(f"✗ Error: {str(e)}")

with col2:
    st.subheader("API Information")
    st.markdown("Get general information about the API")
    
    if st.button("Get API Info", use_container_width=True, type="primary"):
        try:
            with st.spinner("Fetching API info..."):
                response = requests.get(f"{API_BASE_URL}/", timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✓ API Info retrieved successfully")
                    
                    # Display key information
                    if "name" in result:
                        st.markdown(f"**Name:** {result['name']}")
                    if "version" in result:
                        st.markdown(f"**Version:** {result['version']}")
                    if "description" in result:
                        st.markdown(f"**Description:** {result['description']}")
                    
                    # Show full response
                    with st.expander("Full API Information"):
                        st.json(result)
                else:
                    st.error(f"✗ API returned status code: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            st.error("✗ Could not connect to the API")
        except requests.exceptions.Timeout:
            st.error("✗ Request timed out")
        except Exception as e:
            st.error(f"✗ Error: {str(e)}")

st.markdown("---")

# Additional checks for admin
if st.session_state.user_role == "admin":
    st.subheader("Admin: Advanced Diagnostics")
    
    with st.expander("Test API Endpoints"):
        st.markdown("Test various API endpoints to verify functionality")
        
        endpoint = st.text_input("Custom Endpoint", placeholder="/api/v1/...", help="Enter a custom endpoint to test")
        
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
                        st.success("✓ API documentation is accessible")
                        st.info(f"View at: {API_BASE_URL}/docs")
                    else:
                        st.warning("Documentation might not be enabled")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col3:
            if st.button("Test /openapi.json"):
                try:
                    response = requests.get(f"{API_BASE_URL}/openapi.json", timeout=5)
                    if response.status_code == 200:
                        st.success("✓ OpenAPI schema is available")
                        with st.expander("View Schema"):
                            st.json(response.json())
                    else:
                        st.warning("OpenAPI schema might not be enabled")
                except Exception as e:
                    st.error(f"Error: {e}")

# Status indicators
st.markdown("---")
st.subheader("System Requirements Checklist")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Backend Requirements:**")
    st.markdown("""
    - ✓ API server running on configured port
    - ✓ Database connection established
    - ✓ Azure OpenAI credentials configured
    - ✓ Vector store accessible
    """)

with col2:
    st.markdown("**For Testing:**")
    st.markdown("""
    - Research papers ingested into database
    - Embedding model operational
    - LLM endpoints responding
    - Network connectivity stable
    """)

# Footer with tips
st.markdown("---")
st.markdown("### Troubleshooting Tips")

if st.session_state.user_role == "user":
    st.info("""
    **If the API is not responding:**
    1. Contact your administrator
    2. Verify you're on the correct network
    3. Check if scheduled maintenance is in progress
    """)
else:  # admin
    st.info("""
    **Common Issues:**
    1. **Connection Refused**: API server is not running - start it with `uvicorn`
    2. **503 Service Unavailable**: Azure OpenAI not configured - check `.env` file
    3. **Timeout**: Server is overloaded or query is complex - check logs
    4. **404 Not Found**: Endpoint doesn't exist - verify API version and routes
    """)

st.caption("Built for pHera Medical Agent Testing")
