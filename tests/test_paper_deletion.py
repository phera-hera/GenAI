"""
Test Script for Paper Deletion Functionality

This script demonstrates how to use the PaperManager to delete papers
from the database and GCP storage. Use this for testing in your Streamlit
admin panel or as a standalone script.
"""

import asyncio
import uuid

from medical_agent.core.paper_manager import PaperManager, delete_paper_by_id, delete_paper_by_doi
from medical_agent.infrastructure.database.session import get_session_context


async def list_all_papers():
    """List all papers in the database."""
    print("\n" + "=" * 80)
    print("LISTING ALL PAPERS")
    print("=" * 80)

    manager = PaperManager()
    async with get_session_context() as session:
        papers = await manager.list_papers(session, limit=50)

        if not papers:
            print("No papers found in the database.")
            return []

        print(f"\nFound {len(papers)} paper(s):\n")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. ID: {paper['id']}")
            print(f"   Title: {paper['title']}")
            print(f"   Authors: {paper['authors']}")
            print(f"   Year: {paper['publication_year']}")
            print(f"   DOI: {paper['doi']}")
            print(f"   Chunks: {paper['chunk_count']}")
            print(f"   Processed: {paper['is_processed']}")
            print()

        return papers


async def get_paper_details(paper_id: str):
    """Get detailed information about a specific paper."""
    print("\n" + "=" * 80)
    print(f"PAPER DETAILS FOR ID: {paper_id}")
    print("=" * 80)

    manager = PaperManager()
    async with get_session_context() as session:
        paper_info = await manager.get_paper_info(session, uuid.UUID(paper_id))

        if not paper_info:
            print(f"Paper not found: {paper_id}")
            return None

        print(f"\nTitle: {paper_info['title']}")
        print(f"Authors: {paper_info['authors']}")
        print(f"Journal: {paper_info['journal']}")
        print(f"Year: {paper_info['publication_year']}")
        print(f"DOI: {paper_info['doi']}")
        print(f"GCP Path: {paper_info['gcp_path']}")
        print(f"Processed: {paper_info['is_processed']}")
        print(f"Chunks: {paper_info['chunk_count']}")
        print(f"Created: {paper_info['created_at']}")

        return paper_info


async def delete_paper_example(paper_id: str, delete_from_gcp: bool = True):
    """
    Example: Delete a paper by ID.

    Args:
        paper_id: UUID string of the paper to delete
        delete_from_gcp: Whether to also delete from GCP storage
    """
    print("\n" + "=" * 80)
    print(f"DELETING PAPER: {paper_id}")
    print(f"Delete from GCP: {delete_from_gcp}")
    print("=" * 80)

    try:
        async with get_session_context() as session:
            # Get paper info before deletion (for confirmation)
            manager = PaperManager()
            paper_info = await manager.get_paper_info(session, uuid.UUID(paper_id))

            if not paper_info:
                print(f"\nERROR: Paper not found: {paper_id}")
                return

            print(f"\nPaper to delete:")
            print(f"  Title: {paper_info['title']}")
            print(f"  Chunks: {paper_info['chunk_count']}")
            print(f"  GCP Path: {paper_info['gcp_path']}")

            # Confirm deletion
            confirm = input("\nAre you sure you want to delete this paper? (yes/no): ")
            if confirm.lower() != "yes":
                print("Deletion cancelled.")
                return

            # Delete the paper
            result = await delete_paper_by_id(
                session,
                uuid.UUID(paper_id),
                delete_from_gcp=delete_from_gcp,
            )

            # Display results
            print("\n" + "-" * 80)
            print("DELETION RESULT")
            print("-" * 80)
            print(f"Paper: {result.paper_title}")
            print(f"Deleted from DB: {result.deleted_from_db}")
            print(f"Deleted from GCP: {result.deleted_from_gcp}")
            print(f"Chunks deleted: {result.chunks_deleted}")

            if result.success:
                print("\n✓ SUCCESS: Paper fully deleted!")
            elif result.partial_success:
                print(f"\n⚠ PARTIAL SUCCESS: {result.error}")
            else:
                print(f"\n✗ FAILED: {result.error}")

    except ValueError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


async def delete_paper_by_doi_example(doi: str):
    """
    Example: Delete a paper by DOI.

    Args:
        doi: Digital Object Identifier of the paper
    """
    print("\n" + "=" * 80)
    print(f"DELETING PAPER BY DOI: {doi}")
    print("=" * 80)

    try:
        async with get_session_context() as session:
            result = await delete_paper_by_doi(session, doi)

            print("\n" + "-" * 80)
            print("DELETION RESULT")
            print("-" * 80)
            print(f"Paper: {result.paper_title}")
            print(f"Deleted from DB: {result.deleted_from_db}")
            print(f"Deleted from GCP: {result.deleted_from_gcp}")
            print(f"Chunks deleted: {result.chunks_deleted}")

            if result.success:
                print("\n✓ SUCCESS: Paper fully deleted!")

    except ValueError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")


async def bulk_delete_example(paper_ids: list[str]):
    """
    Example: Delete multiple papers at once.

    Args:
        paper_ids: List of paper UUID strings
    """
    print("\n" + "=" * 80)
    print(f"BULK DELETING {len(paper_ids)} PAPERS")
    print("=" * 80)

    manager = PaperManager()

    try:
        async with get_session_context() as session:
            # Convert string IDs to UUIDs
            uuids = [uuid.UUID(pid) for pid in paper_ids]

            # Confirm
            print(f"\nAbout to delete {len(uuids)} papers:")
            for pid in paper_ids:
                print(f"  - {pid}")

            confirm = input("\nProceed with bulk deletion? (yes/no): ")
            if confirm.lower() != "yes":
                print("Bulk deletion cancelled.")
                return

            # Bulk delete
            results = await manager.delete_papers_bulk(session, uuids)

            # Display results
            print("\n" + "-" * 80)
            print("BULK DELETION RESULTS")
            print("-" * 80)

            success_count = sum(1 for r in results if r.success)
            partial_count = sum(1 for r in results if r.partial_success and not r.success)
            failed_count = len(results) - success_count - partial_count

            print(f"\nTotal: {len(results)}")
            print(f"Success: {success_count}")
            print(f"Partial: {partial_count}")
            print(f"Failed: {failed_count}")

            print("\nDetails:")
            for result in results:
                status = "✓" if result.success else ("⚠" if result.partial_success else "✗")
                print(f"{status} {result.paper_title} - Chunks: {result.chunks_deleted}")
                if result.error:
                    print(f"   Error: {result.error}")

    except Exception as e:
        print(f"\nBULK DELETION FAILED: {e}")
        import traceback
        traceback.print_exc()


async def interactive_mode():
    """Interactive mode for testing paper deletion."""
    print("\n" + "=" * 80)
    print("PAPER DELETION - INTERACTIVE MODE")
    print("=" * 80)

    while True:
        print("\nOptions:")
        print("1. List all papers")
        print("2. Get paper details")
        print("3. Delete paper by ID")
        print("4. Delete paper by DOI")
        print("5. Bulk delete papers")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            await list_all_papers()

        elif choice == "2":
            paper_id = input("Enter paper ID: ").strip()
            await get_paper_details(paper_id)

        elif choice == "3":
            paper_id = input("Enter paper ID: ").strip()
            delete_gcp = input("Delete from GCP? (yes/no, default: yes): ").strip().lower()
            delete_gcp = delete_gcp != "no"
            await delete_paper_example(paper_id, delete_from_gcp=delete_gcp)

        elif choice == "4":
            doi = input("Enter DOI: ").strip()
            await delete_paper_by_doi_example(doi)

        elif choice == "5":
            ids_input = input("Enter paper IDs (comma-separated): ").strip()
            paper_ids = [pid.strip() for pid in ids_input.split(",")]
            await bulk_delete_example(paper_ids)

        elif choice == "6":
            print("\nExiting...")
            break

        else:
            print("\nInvalid choice. Please try again.")


async def main():
    """Main entry point for testing."""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            await list_all_papers()

        elif command == "details" and len(sys.argv) > 2:
            paper_id = sys.argv[2]
            await get_paper_details(paper_id)

        elif command == "delete" and len(sys.argv) > 2:
            paper_id = sys.argv[2]
            delete_gcp = "--no-gcp" not in sys.argv
            await delete_paper_example(paper_id, delete_from_gcp=delete_gcp)

        elif command == "delete-doi" and len(sys.argv) > 2:
            doi = sys.argv[2]
            await delete_paper_by_doi_example(doi)

        elif command == "interactive":
            await interactive_mode()

        else:
            print("Usage:")
            print("  python test_paper_deletion.py list")
            print("  python test_paper_deletion.py details <paper_id>")
            print("  python test_paper_deletion.py delete <paper_id> [--no-gcp]")
            print("  python test_paper_deletion.py delete-doi <doi>")
            print("  python test_paper_deletion.py interactive")
    else:
        # Interactive mode by default
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
