"""
Paper Management Page

ADMIN ONLY - Manage research papers in the system.
"""

import asyncio
import uuid
import streamlit as st

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


# Main UI
st.title("Paper Management")
st.markdown("Manage research papers in the medical agent database")
st.markdown("---")

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["View All Papers", "Delete Paper", "Bulk Operations"])

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

# Footer
st.markdown("---")
st.caption("⚠️ Admin Area - All operations are logged and monitored")
