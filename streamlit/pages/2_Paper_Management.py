"""
Paper Management Page

ADMIN ONLY - Manage research papers in the system.
"""

import asyncio
import uuid
import streamlit as st
import requests
from typing import List, Dict, Any

try:
    from medical_agent.core.paper_manager import PaperManager
    from medical_agent.infrastructure.database.session import get_session_context
    PAPER_MANAGER_AVAILABLE = True
except ImportError:
    PAPER_MANAGER_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Paper Management",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check if user is admin
if "user_role" not in st.session_state or st.session_state.user_role != "admin":
    st.error("Access Denied: This page is only available to administrators.")
    st.info("Please switch to admin mode from the Home page.")
    st.stop()

# Check if paper manager is available
if not PAPER_MANAGER_AVAILABLE:
    st.error("Paper Management module is not available. Please ensure the medical_agent package is properly installed.")
    st.stop()

# Helper Functions
async def fetch_all_papers():
    """Fetch all papers from the database."""
    manager = PaperManager()
    async with get_session_context() as session:
        return await manager.list_papers(session, limit=100)


async def fetch_paper_details(paper_id: str):
    """Fetch detailed information about a paper."""
    manager = PaperManager()
    async with get_session_context() as session:
        return await manager.get_paper_info(session, uuid.UUID(paper_id))


async def delete_paper(paper_id: str, delete_from_gcp: bool = True):
    """Delete a paper and return the result."""
    manager = PaperManager()
    async with get_session_context() as session:
        return await manager.delete_paper(
            session,
            uuid.UUID(paper_id),
            delete_from_gcp=delete_from_gcp,
        )


# Professional styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
    }

    body, p, div, span, label {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
    }

    h1, h2, h3 {
        color: #0F5257 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
    }

    .success-box {
        background: #1A4D2E; /* Forest Green bg */
        padding: 0.75rem;
        border-radius: 4px;
        border-left: 4px solid #2D7A4A;
        margin: 0.5rem 0;
        color: #E8F5E9;
    }
    .info-box {
        background: #0D3D4D;
        padding: 0.75rem;
        border-radius: 4px;
        border-left: 4px solid #0F5257;
        margin: 0.5rem 0;
        color: #E1F5FE;
    }
</style>
""", unsafe_allow_html=True)

st.title("Paper Management")
st.markdown("---")

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["View All Papers", "Delete Paper", "Bulk Operations", "Ingest Papers"])

# TAB 1: View All Papers
with tab1:
    st.header("All Papers in Database")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    
    # Fetch papers
    try:
        papers = asyncio.run(fetch_all_papers())
        
        if not papers:
            st.info("No papers found in the database.")
        else:
            st.success(f"Found {len(papers)} paper(s) in the database")
            
            # Display papers in a table
            for i, paper in enumerate(papers):
                with st.expander(f"Paper {i+1}: {paper['title'][:80]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**ID:** `{paper['id']}`")
                        st.markdown(f"**Title:** {paper['title']}")
                        st.markdown(f"**Authors:** {paper['authors']}")
                        st.markdown(f"**Year:** {paper.get('publication_year', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**DOI:** {paper.get('doi', 'N/A')}")
                        st.markdown(f"**Chunks:** {paper['chunk_count']}")
                        st.markdown(f"**Processed:** {paper['is_processed']}")
                        st.markdown(f"**Created:** {paper.get('created_at', 'N/A')}")
                    
                    # Quick delete button
                    if st.button(f"Delete This Paper", key=f"delete_{paper['id']}"):
                        st.session_state[f"delete_confirm_{paper['id']}"] = True
                    
                    # Confirmation dialog
                    if st.session_state.get(f"delete_confirm_{paper['id']}", False):
                        st.warning(f"Are you sure you want to delete '{paper['title']}'?")
                        st.error("This action cannot be undone!")
                        
                        col_yes, col_no, col_space = st.columns([1, 1, 2])
                        
                        with col_yes:
                            if st.button("Yes, Delete", key=f"confirm_yes_{paper['id']}", type="primary"):
                                with st.spinner("Deleting paper..."):
                                    result = asyncio.run(delete_paper(paper['id']))
                                    
                                    if result.success:
                                        st.success(f"Deleted '{result.paper_title}' and {result.chunks_deleted} chunks")
                                        st.session_state[f"delete_confirm_{paper['id']}"] = False
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {result.error}")
                        
                        with col_no:
                            if st.button("Cancel", key=f"confirm_no_{paper['id']}"):
                                st.session_state[f"delete_confirm_{paper['id']}"] = False
                                st.rerun()
    
    except Exception as e:
        st.error(f"Error fetching papers: {str(e)}")

# TAB 2: Delete Specific Paper
with tab2:
    st.header("Delete Specific Paper")
    
    paper_id = st.text_input(
        "Enter Paper ID (UUID)",
        placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000",
        help="You can find paper IDs in the 'View All Papers' tab"
    )
    
    delete_from_gcp = st.checkbox(
        "Also delete PDF from GCP Storage",
        value=True,
        help="Uncheck if you want to keep the PDF in cloud storage"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        preview_button = st.button("Preview Paper", use_container_width=True)
    
    if preview_button:
        if not paper_id:
            st.error("Please enter a Paper ID")
        else:
            try:
                with st.spinner("Loading paper details..."):
                    paper_info = asyncio.run(fetch_paper_details(paper_id))
                    
                    if not paper_info:
                        st.error(f"Paper not found: {paper_id}")
                    else:
                        st.success("Paper found!")
                        
                        # Display paper details in a nice format
                        st.markdown("---")
                        st.subheader("Paper Details")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Title:** {paper_info['title']}")
                            st.markdown(f"**Authors:** {paper_info['authors']}")
                            st.markdown(f"**Year:** {paper_info.get('publication_year', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**DOI:** {paper_info.get('doi', 'N/A')}")
                            st.markdown(f"**Chunks:** {paper_info['chunk_count']}")
                            st.markdown(f"**ID:** `{paper_info['id']}`")
                        
                        # Store in session state for deletion
                        st.session_state["paper_to_delete"] = paper_info
            
            except ValueError as e:
                st.error(f"Invalid Paper ID format: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Show delete button if paper is loaded
    if "paper_to_delete" in st.session_state:
        paper_info = st.session_state["paper_to_delete"]
        
        st.markdown("---")
        st.warning(f"⚠️ **You are about to delete:** {paper_info['title']}")
        st.error(f"This will permanently delete {paper_info['chunk_count']} chunks and the paper record.")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("DELETE PAPER", type="primary", use_container_width=True):
                try:
                    with st.spinner("Deleting paper..."):
                        result = asyncio.run(delete_paper(paper_id, delete_from_gcp))
                        
                        if result.success:
                            st.success(f"Successfully deleted '{result.paper_title}'")
                            st.info(f"Deleted {result.chunks_deleted} chunks from database")
                            if result.deleted_from_gcp:
                                st.info("Deleted PDF from GCP Storage")
                            
                            # Clear session state
                            del st.session_state["paper_to_delete"]
                            st.balloons()
                        
                        elif result.partial_success:
                            st.warning(f"Partial deletion: {result.error}")
                            st.info(f"DB deleted: {result.deleted_from_db}")
                            st.info(f"GCP deleted: {result.deleted_from_gcp}")
                        
                        else:
                            st.error(f"Deletion failed: {result.error}")
                
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                del st.session_state["paper_to_delete"]
                st.rerun()

# TAB 3: Bulk Operations
with tab3:
    st.header("Bulk Operations")
    st.warning("⚠️ **Warning**: Bulk deletion is a powerful operation. Use with extreme caution!")
    
    # Multi-select for papers
    try:
        papers = asyncio.run(fetch_all_papers())
        
        if not papers:
            st.info("No papers available for bulk operations.")
        else:
            # Create a mapping for display
            paper_options = {
                f"{p['title'][:60]}... ({p.get('publication_year', 'N/A')}) - {p['chunk_count']} chunks": p['id']
                for p in papers
            }
            
            selected_papers = st.multiselect(
                "Select papers to delete:",
                options=list(paper_options.keys()),
                help="You can select multiple papers"
            )
            
            if selected_papers:
                st.info(f"Selected {len(selected_papers)} paper(s) for deletion")
                
                delete_from_gcp_bulk = st.checkbox(
                    "Also delete PDFs from GCP Storage",
                    value=True,
                    key="bulk_delete_gcp",
                )
                
                st.markdown("---")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if st.button("DELETE SELECTED", type="primary", use_container_width=True):
                        st.session_state["bulk_confirm"] = True
                
                if st.session_state.get("bulk_confirm", False):
                    st.error("⚠️ **FINAL WARNING**: This will permanently delete all selected papers!")
                    
                    col_confirm, col_cancel, col_space = st.columns([1, 1, 2])
                    
                    with col_confirm:
                        if st.button("CONFIRM DELETION", key="bulk_final_confirm"):
                            try:
                                with st.spinner(f"Deleting {len(selected_papers)} papers..."):
                                    # Get paper IDs
                                    paper_ids = [paper_options[title] for title in selected_papers]
                                    
                                    # Delete papers one by one
                                    results = []
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    for i, paper_id in enumerate(paper_ids):
                                        status_text.text(f"Deleting paper {i+1} of {len(paper_ids)}...")
                                        result = asyncio.run(delete_paper(paper_id, delete_from_gcp_bulk))
                                        results.append(result)
                                        progress_bar.progress((i + 1) / len(paper_ids))
                                    
                                    # Display results
                                    success_count = sum(1 for r in results if r.success)
                                    failed_count = len(results) - success_count
                                    
                                    st.success(f"✓ Successfully deleted {success_count} paper(s)")
                                    
                                    if failed_count > 0:
                                        st.error(f"✗ Failed to delete {failed_count} paper(s)")
                                    
                                    # Show details
                                    with st.expander("View Detailed Results"):
                                        for result in results:
                                            status = "✓" if result.success else "✗"
                                            st.write(f"{status} {result.paper_title} - {result.chunks_deleted} chunks")
                                            if result.error:
                                                st.write(f"   Error: {result.error}")
                                    
                                    # Clear confirmation state
                                    st.session_state["bulk_confirm"] = False
                                    
                                    if failed_count == 0:
                                        st.balloons()
                            
                            except Exception as e:
                                st.error(f"Bulk deletion failed: {e}")
                    
                    with col_cancel:
                        if st.button("Cancel", key="bulk_cancel"):
                            st.session_state["bulk_confirm"] = False
                            st.rerun()
    
    except Exception as e:
        st.error(f"Error loading papers: {str(e)}")

# TAB 4: Ingest Papers
with tab4:
    st.header("Ingest Papers from GCP")

    API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

    st.markdown("""
    Upload PDFs to your GCP bucket, then use this interface to ingest them into the database.
    The system will parse PDFs, extract metadata, chunk content, and create embeddings.
    """)

    st.markdown("---")

    # Section 1: List Available Papers
    st.subheader("1. List Papers from GCP Bucket")

    col1, col2 = st.columns([1, 3])
    with col1:
        list_button = st.button("List Papers from GCP", use_container_width=True, type="primary")

    if list_button:
        try:
            with st.spinner("Fetching papers from GCP bucket..."):
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/ingest",
                    json={"list_bucket": True},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    available_papers = result.get("available_papers", [])

                    if not available_papers:
                        st.info("No papers found in GCP bucket")
                    else:
                        # Get currently ingested papers
                        try:
                            ingested_papers = asyncio.run(fetch_all_papers())
                            ingested_titles = {p['title'] for p in ingested_papers}
                        except:
                            ingested_titles = set()

                        # Categorize papers
                        new_papers = []
                        existing_papers = []

                        for paper in available_papers:
                            paper_name = paper.get("name", "")
                            # Simple check - just filename matching
                            if any(paper_name in title or title in paper_name for title in ingested_titles):
                                existing_papers.append(paper)
                            else:
                                new_papers.append(paper)

                        st.success(f"Found {len(available_papers)} paper(s) in GCP bucket")

                        # Display new papers
                        if new_papers:
                            st.markdown(f"**New Papers ({len(new_papers)})** - Not yet ingested")
                            new_paper_data = []
                            for p in new_papers:
                                new_paper_data.append({
                                    "Filename": p.get("name", ""),
                                    "Size": f"{p.get('size', 0) / 1024:.1f} KB",
                                    "Status": "✓ Ready to ingest"
                                })
                            st.dataframe(new_paper_data, use_container_width=True)

                        # Display already ingested
                        if existing_papers:
                            with st.expander(f"Already Ingested ({len(existing_papers)})"):
                                existing_paper_data = []
                                for p in existing_papers:
                                    existing_paper_data.append({
                                        "Filename": p.get("name", ""),
                                        "Size": f"{p.get('size', 0) / 1024:.1f} KB",
                                        "Status": "Already in database"
                                    })
                                st.dataframe(existing_paper_data, use_container_width=True)

                        # Store papers in session state for ingestion
                        st.session_state["available_papers"] = available_papers
                        st.session_state["new_papers"] = [p.get("path") for p in new_papers]

                else:
                    st.error(f"API Error: {response.status_code}")
                    try:
                        st.json(response.json())
                    except:
                        st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API")
        except requests.exceptions.Timeout:
            st.error("Timeout Error: Request took too long")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Section 2: Ingest Papers
    if "available_papers" in st.session_state and st.session_state.get("new_papers"):
        st.markdown("---")
        st.subheader("2. Ingest Papers")

        available_papers = st.session_state["available_papers"]
        new_papers = st.session_state["new_papers"]

        # Option 1: Ingest all new papers
        st.markdown("**Option 1: Ingest All New Papers**")
        col1, col2 = st.columns([1, 3])
        with col1:
            ingest_all_button = st.button(
                f"Ingest All ({len(new_papers)} papers)",
                use_container_width=True,
                type="primary"
            )

        # Option 2: Select specific papers
        st.markdown("**Option 2: Select Specific Papers**")
        paper_options = {p.get("name", ""): p.get("path", "") for p in available_papers}

        selected_papers = st.multiselect(
            "Select papers to ingest:",
            options=list(paper_options.keys())
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            ingest_selected_button = st.button(
                f"Ingest Selected ({len(selected_papers)} papers)",
                use_container_width=True,
                disabled=len(selected_papers) == 0
            )

        # Dry run option
        dry_run = st.checkbox(
            "Dry Run (validate without storing)",
            help="Test ingestion without saving to database"
        )

        # Process ingestion
        if ingest_all_button or ingest_selected_button:
            # Determine which papers to ingest
            if ingest_all_button:
                paths_to_ingest = new_papers
            else:
                paths_to_ingest = [paper_options[name] for name in selected_papers]

            st.markdown("---")
            st.subheader("Ingestion Progress")

            try:
                # Prepare request
                request_data = {
                    "gcp_paths": paths_to_ingest,
                    "dry_run": dry_run
                }

                # Show request payload for debugging
                with st.expander("Request Payload (Admin Debug)"):
                    st.json(request_data)

                # Make API call
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Starting ingestion...")

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/ingest",
                    json=request_data,
                    timeout=300  # 5 minutes timeout
                )

                progress_bar.progress(100)

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("Ingestion Complete!" if not dry_run else "Dry Run Complete!")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Papers Processed", len(result.get("results", [])))
                    with col2:
                        success_count = sum(1 for r in result.get("results", []) if r.get("success"))
                        st.metric("Successful", success_count)
                    with col3:
                        total_time = result.get("total_processing_time_ms", 0)
                        st.metric("Total Time", f"{total_time:.0f} ms")

                    # Detailed results
                    st.markdown("---")
                    st.markdown("### Detailed Results")

                    for i, paper_result in enumerate(result.get("results", []), 1):
                        paper_path = paper_result.get("paper_path", "Unknown")
                        success = paper_result.get("success", False)

                        status_icon = "✓" if success else "✗"
                        status_color = "#2D7A4A" if success else "#DC2626"

                        with st.expander(f"{status_icon} Paper {i}: {paper_path.split('/')[-1]}"):
                            if success:
                                st.markdown(f"<div style='color: {status_color}; font-weight: bold;'>SUCCESS</div>", unsafe_allow_html=True)

                                # Paper details
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Paper ID:** `{paper_result.get('paper_id', 'N/A')}`")
                                    st.markdown(f"**Chunks Created:** {paper_result.get('chunks_count', 0)}")
                                    st.markdown(f"**Duration:** {paper_result.get('duration_ms', 0):.0f} ms")

                                with col2:
                                    # Metadata counts
                                    metadata = paper_result.get("metadata_extracted", {})
                                    st.markdown("**Metadata Extracted:**")
                                    st.markdown(f"- Ethnicities: {len(metadata.get('ethnicities', []))}")
                                    st.markdown(f"- Diagnoses: {len(metadata.get('diagnoses', []))}")
                                    st.markdown(f"- Symptoms: {len(metadata.get('symptoms', []))}")
                                    st.markdown(f"- Hormone Therapy: {len(metadata.get('hormone_therapy', []))}")
                                    st.markdown(f"- Birth Control: {len(metadata.get('birth_control', []))}")

                                # Stage breakdown
                                stages = paper_result.get("stages", {})
                                if stages:
                                    st.markdown("**Stage Breakdown:**")
                                    stage_data = []
                                    for stage_name, stage_info in stages.items():
                                        stage_data.append({
                                            "Stage": stage_name.replace("_", " ").title(),
                                            "Duration (ms)": f"{stage_info.get('duration_ms', 0):.0f}",
                                            "Status": "✓ Complete"
                                        })
                                    st.dataframe(stage_data, use_container_width=True)

                            else:
                                st.markdown(f"<div style='color: {status_color}; font-weight: bold;'>FAILED</div>", unsafe_allow_html=True)
                                error_msg = paper_result.get("error", "Unknown error")
                                st.error(f"Error: {error_msg}")

                    # Full response for admin debugging
                    with st.expander("Full API Response (Admin Debug)"):
                        st.json(result)

                    if not dry_run and success_count > 0:
                        st.balloons()

                elif response.status_code == 503:
                    st.error("Service Unavailable: The ingestion service is not ready. Make sure all dependencies are installed.")
                elif response.status_code == 400:
                    st.error("Bad Request: Invalid request parameters")
                    try:
                        st.json(response.json())
                    except:
                        st.text(response.text)
                else:
                    st.error(f"API Error: {response.status_code}")
                    try:
                        st.json(response.json())
                    except:
                        st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the API")
            except requests.exceptions.Timeout:
                st.error("Timeout Error: Ingestion took too long (>5 minutes)")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.caption("⚠️ Admin Area - All operations are logged and monitored")
