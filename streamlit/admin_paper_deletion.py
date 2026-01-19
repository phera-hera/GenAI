"""
Streamlit Admin Panel - Paper Deletion

Example implementation of paper deletion UI for your admin panel.
You can integrate this into your main Streamlit app later.
"""

import asyncio
import uuid

import streamlit as st

from medical_agent.core.paper_manager import PaperManager
from medical_agent.infrastructure.database.session import get_session_context


# =============================================================================
# Helper Functions
# =============================================================================


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


# =============================================================================
# Streamlit UI
# =============================================================================


def main():
    st.set_page_config(
        page_title="Paper Management Admin",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 Paper Management - Admin Panel")
    st.markdown("---")

    # Sidebar navigation
    menu = st.sidebar.radio(
        "Navigation",
        ["View All Papers", "Delete Paper", "Bulk Operations"],
    )

    if menu == "View All Papers":
        show_all_papers()
    elif menu == "Delete Paper":
        show_delete_paper()
    elif menu == "Bulk Operations":
        show_bulk_operations()


def show_all_papers():
    """Display all papers in the database."""
    st.header("All Papers")

    if st.button("🔄 Refresh"):
        st.rerun()

    # Fetch papers
    papers = asyncio.run(fetch_all_papers())

    if not papers:
        st.info("No papers found in the database.")
        return

    st.success(f"Found {len(papers)} paper(s)")

    # Display papers in a table
    for i, paper in enumerate(papers):
        with st.expander(f"📑 {paper['title'][:100]}..."):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**ID:** `{paper['id']}`")
                st.markdown(f"**Title:** {paper['title']}")
                st.markdown(f"**Authors:** {paper['authors']}")
                st.markdown(f"**Year:** {paper['publication_year']}")

            with col2:
                st.markdown(f"**DOI:** {paper['doi']}")
                st.markdown(f"**Chunks:** {paper['chunk_count']}")
                st.markdown(f"**Processed:** {paper['is_processed']}")
                st.markdown(f"**Created:** {paper['created_at']}")

            # Quick delete button
            if st.button(f"🗑️ Delete", key=f"delete_{paper['id']}"):
                st.session_state[f"delete_confirm_{paper['id']}"] = True

            # Confirmation dialog
            if st.session_state.get(f"delete_confirm_{paper['id']}", False):
                st.warning(f"⚠️ Are you sure you want to delete '{paper['title']}'?")

                col_yes, col_no = st.columns(2)

                with col_yes:
                    if st.button("✓ Yes, Delete", key=f"confirm_yes_{paper['id']}"):
                        with st.spinner("Deleting..."):
                            result = asyncio.run(delete_paper(paper['id']))

                            if result.success:
                                st.success(f"✓ Deleted '{result.paper_title}' and {result.chunks_deleted} chunks")
                                st.session_state[f"delete_confirm_{paper['id']}"] = False
                                st.rerun()
                            else:
                                st.error(f"✗ Failed: {result.error}")

                with col_no:
                    if st.button("✗ Cancel", key=f"confirm_no_{paper['id']}"):
                        st.session_state[f"delete_confirm_{paper['id']}"] = False
                        st.rerun()


def show_delete_paper():
    """UI for deleting a specific paper by ID or DOI."""
    st.header("Delete Specific Paper")

    # Tab selection
    tab1, tab2 = st.tabs(["By Paper ID", "By DOI"])

    with tab1:
        st.subheader("Delete by Paper ID")

        paper_id = st.text_input(
            "Enter Paper ID (UUID)",
            placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000",
            key="delete_by_id_input",
        )

        delete_from_gcp = st.checkbox(
            "Also delete PDF from GCP Storage",
            value=True,
            key="delete_gcp_checkbox",
        )

        if st.button("🔍 Preview Paper", key="preview_button"):
            if not paper_id:
                st.error("Please enter a Paper ID")
            else:
                with st.spinner("Loading paper details..."):
                    try:
                        paper_info = asyncio.run(fetch_paper_details(paper_id))

                        if not paper_info:
                            st.error(f"Paper not found: {paper_id}")
                        else:
                            st.success("Paper found!")

                            # Display paper details
                            st.json(paper_info)

                            # Store in session state for deletion
                            st.session_state["paper_to_delete"] = paper_info

                    except ValueError as e:
                        st.error(f"Invalid Paper ID: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Show delete button if paper is loaded
        if "paper_to_delete" in st.session_state:
            paper_info = st.session_state["paper_to_delete"]

            st.markdown("---")
            st.warning(f"⚠️ You are about to delete: **{paper_info['title']}**")
            st.info(f"This will delete {paper_info['chunk_count']} chunks and the paper record.")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🗑️ DELETE PAPER", type="primary", key="final_delete_button"):
                    with st.spinner("Deleting paper..."):
                        try:
                            result = asyncio.run(delete_paper(paper_id, delete_from_gcp))

                            if result.success:
                                st.success(f"✓ Successfully deleted '{result.paper_title}'")
                                st.info(f"Deleted {result.chunks_deleted} chunks from database")
                                if result.deleted_from_gcp:
                                    st.info("Deleted PDF from GCP Storage")

                                # Clear session state
                                del st.session_state["paper_to_delete"]

                            elif result.partial_success:
                                st.warning(f"⚠️ Partial deletion: {result.error}")
                                st.info(f"DB deleted: {result.deleted_from_db}")
                                st.info(f"GCP deleted: {result.deleted_from_gcp}")

                            else:
                                st.error(f"✗ Deletion failed: {result.error}")

                        except Exception as e:
                            st.error(f"Unexpected error: {e}")

            with col2:
                if st.button("Cancel", key="cancel_delete_button"):
                    del st.session_state["paper_to_delete"]
                    st.rerun()

    with tab2:
        st.subheader("Delete by DOI")
        st.info("Coming soon! You can implement this similarly to 'By Paper ID'")


def show_bulk_operations():
    """UI for bulk paper operations."""
    st.header("Bulk Operations")

    st.warning("⚠️ Bulk deletion is a powerful operation. Use with caution!")

    # Multi-select for papers
    papers = asyncio.run(fetch_all_papers())

    if not papers:
        st.info("No papers available for bulk operations.")
        return

    # Create a mapping for display
    paper_options = {
        f"{p['title'][:80]} ({p['publication_year']}) - {p['chunk_count']} chunks": p['id']
        for p in papers
    }

    selected_papers = st.multiselect(
        "Select papers to delete:",
        options=list(paper_options.keys()),
        key="bulk_select",
    )

    if selected_papers:
        st.info(f"Selected {len(selected_papers)} paper(s)")

        delete_from_gcp = st.checkbox(
            "Also delete PDFs from GCP Storage",
            value=True,
            key="bulk_delete_gcp",
        )

        if st.button("🗑️ DELETE SELECTED PAPERS", type="primary", key="bulk_delete_button"):
            st.warning("⚠️ This will permanently delete all selected papers!")

            if st.button("Confirm Bulk Deletion", key="bulk_confirm"):
                with st.spinner(f"Deleting {len(selected_papers)} papers..."):
                    try:
                        # Get paper IDs
                        paper_ids = [paper_options[title] for title in selected_papers]

                        # Delete papers one by one (or use bulk method)
                        results = []
                        progress_bar = st.progress(0)

                        for i, paper_id in enumerate(paper_ids):
                            result = asyncio.run(delete_paper(paper_id, delete_from_gcp))
                            results.append(result)
                            progress_bar.progress((i + 1) / len(paper_ids))

                        # Display results
                        success_count = sum(1 for r in results if r.success)
                        failed_count = len(results) - success_count

                        st.success(f"✓ Deleted {success_count} paper(s)")

                        if failed_count > 0:
                            st.error(f"✗ Failed to delete {failed_count} paper(s)")

                        # Show details
                        with st.expander("View Details"):
                            for result in results:
                                status = "✓" if result.success else "✗"
                                st.write(f"{status} {result.paper_title} - {result.chunks_deleted} chunks")
                                if result.error:
                                    st.write(f"   Error: {result.error}")

                        st.rerun()

                    except Exception as e:
                        st.error(f"Bulk deletion failed: {e}")


# =============================================================================
# Run the app
# =============================================================================

if __name__ == "__main__":
    main()
