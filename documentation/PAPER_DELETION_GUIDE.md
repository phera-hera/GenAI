# Paper Deletion System - Documentation

# fast intractive command
python3 tests/test_paper_deletion.py interactive


## Overview

A complete paper deletion system that removes medical research papers from:
- **PostgreSQL database** (paper metadata + chunks)
- **pgvector embeddings** (automatically with chunks)
- **GCP Cloud Storage** (PDF files)

**Query logs are preserved** for audit trail with citations intact.

---

## What Was Built

### 1. Core Service: `PaperManager`
**Location:** `src/medical_agent/core/paper_manager.py`

Main class that handles all paper deletion logic with proper error handling and transaction management.

**Key Features:**
- Delete papers by ID or DOI
- Bulk deletion support
- Coordinated database + GCP cleanup
- Detailed deletion results
- Query log preservation
- Transaction rollback on errors

### 2. Test Script
**Location:** `test_paper_deletion.py`

Interactive CLI tool for testing deletion functionality before integrating into your app.

**Modes:**
- List all papers
- Get paper details
- Delete by ID
- Delete by DOI
- Bulk delete
- Interactive mode

### 3. Streamlit Example
**Location:** `streamlit/admin_paper_deletion.py`

Ready-to-use Streamlit admin panel UI that you can integrate later.

**Features:**
- View all papers
- Delete single paper with confirmation
- Bulk delete with multi-select
- Preview before deletion
- Real-time progress indicators

---

## How It Works

### Deletion Flow

```
User triggers deletion
       ↓
[1] Fetch paper from PostgreSQL
       ↓
[2] Count chunks (for reporting)
       ↓
[3] Delete paper from database
       ↓ (CASCADE triggers)
[4] Chunks auto-delete
       ↓ (Embeddings in chunks)
[5] Embeddings auto-delete
       ↓
[6] Delete PDF from GCP
       ↓
[7] Return detailed result
       ↓
Query logs stay intact ✓
```

### What Gets Deleted

✅ **Paper record** (from `papers` table)
✅ **All chunks** (from `paper_chunks` table - cascade)
✅ **All embeddings** (stored in chunk records)
✅ **PDF file** (from GCP Cloud Storage)

### What Stays

✓ **Query logs** (preserved with citations for audit)
✓ **User data** (health profiles, etc.)

---

## Quick Start

### Option 1: Test with CLI Script

```bash
# List all papers
python test_paper_deletion.py list

# Get paper details
python test_paper_deletion.py details <paper_id>

# Delete a paper (with GCP)
python test_paper_deletion.py delete <paper_id>

# Delete without removing from GCP
python test_paper_deletion.py delete <paper_id> --no-gcp

# Interactive mode
python test_paper_deletion.py interactive
```

### Option 2: Use in Your Code

```python
import asyncio
import uuid
from medical_agent.core.paper_manager import PaperManager
from medical_agent.infrastructure.database.session import get_session_context

async def delete_my_paper():
    manager = PaperManager()

    async with get_session_context() as session:
        # Delete paper
        result = await manager.delete_paper(
            session,
            paper_id=uuid.UUID("your-paper-id-here"),
            delete_from_gcp=True,
        )

        # Check result
        if result.success:
            print(f"✓ Deleted '{result.paper_title}'")
            print(f"  Chunks removed: {result.chunks_deleted}")
        else:
            print(f"✗ Error: {result.error}")

asyncio.run(delete_my_paper())
```

### Option 3: Convenience Functions

```python
from medical_agent.core.paper_manager import delete_paper_by_id, delete_paper_by_doi
from medical_agent.infrastructure.database.session import get_session_context

# Delete by ID
async with get_session_context() as session:
    result = await delete_paper_by_id(session, paper_id)

# Delete by DOI
async with get_session_context() as session:
    result = await delete_paper_by_doi(session, "10.1234/example")
```

---

## API Reference

### `PaperManager` Class

#### `delete_paper(session, paper_id, delete_from_gcp=True, commit=True)`

Delete a single paper.

**Parameters:**
- `session` (AsyncSession): Database session
- `paper_id` (UUID): Paper to delete
- `delete_from_gcp` (bool): Also delete PDF from GCP (default: True)
- `commit` (bool): Commit transaction (default: True)

**Returns:** `DeletionResult` with:
- `paper_id`: UUID of deleted paper
- `paper_title`: Title of deleted paper
- `deleted_from_db`: Whether DB deletion succeeded
- `deleted_from_gcp`: Whether GCP deletion succeeded
- `chunks_deleted`: Number of chunks removed
- `success`: True if both DB and GCP succeeded
- `error`: Error message if any

**Raises:**
- `ValueError`: Paper not found
- `StorageError`: GCP deletion failed (DB still deleted)

---

#### `delete_papers_bulk(session, paper_ids, delete_from_gcp=True)`

Delete multiple papers in one transaction.

**Parameters:**
- `session` (AsyncSession): Database session
- `paper_ids` (list[UUID]): Papers to delete
- `delete_from_gcp` (bool): Also delete PDFs from GCP

**Returns:** `list[DeletionResult]`

**Note:** All-or-nothing transaction. If any deletion fails, entire operation rolls back.

---

#### `get_paper_info(session, paper_id)`

Get paper metadata before deletion.

**Returns:** Dict with paper details or None if not found.

---

#### `list_papers(session, limit=100, offset=0)`

List all papers with pagination.

**Returns:** List of paper info dicts.

---

## Deletion Result Object

```python
@dataclass
class DeletionResult:
    paper_id: uuid.UUID
    paper_title: str
    deleted_from_db: bool
    deleted_from_gcp: bool
    chunks_deleted: int
    gcp_path: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """True if both DB and GCP deletion succeeded"""

    @property
    def partial_success(self) -> bool:
        """True if at least one succeeded"""
```

---

## Integration with Streamlit

When you're ready to add this to your Streamlit admin panel:

1. Copy `streamlit/admin_paper_deletion.py` into your Streamlit app
2. Add it as a new page or section in your admin interface
3. Customize the UI styling to match your app

**Example integration:**

```python
# In your main streamlit app
import streamlit as st
from streamlit.admin_paper_deletion import show_delete_paper

# Add to your admin menu
if st.session_state.get("is_admin"):
    menu = st.sidebar.selectbox("Admin", ["Users", "Papers", "Settings"])

    if menu == "Papers":
        show_delete_paper()
```

---

## Error Handling

### Scenario 1: Paper Not Found
```python
try:
    result = await manager.delete_paper(session, paper_id)
except ValueError as e:
    print(f"Paper doesn't exist: {e}")
```

### Scenario 2: Database Deleted, GCP Failed
```python
result = await manager.delete_paper(session, paper_id)

if result.partial_success and not result.success:
    print(f"Warning: Paper deleted from DB but GCP failed")
    print(f"Error: {result.error}")
    print(f"Orphaned file at: {result.gcp_path}")
    # You may want to log this for manual cleanup
```

### Scenario 3: Transaction Rollback
```python
try:
    result = await manager.delete_paper(session, paper_id)
except Exception as e:
    # Transaction automatically rolled back
    print(f"Deletion failed, no changes made: {e}")
```

---

## Query Log Preservation

**Why query logs are preserved:**

Your `QueryLog` table stores:
- `retrieved_chunk_ids` (JSONB array) - References to chunks
- `citations` (JSONB) - Paper metadata snapshot

When a paper is deleted:
- ✓ `citations` remain intact (historical record)
- ✓ `retrieved_chunk_ids` contain invalid UUIDs (safe, no FK constraint)
- ✓ Audit trail preserved

**Example query log after deletion:**
```json
{
  "query_text": "What causes infections?",
  "retrieved_chunk_ids": ["abc-123", "def-456"],  // May be deleted
  "citations": {
    "title": "Bacterial Infections Study",  // Still here!
    "authors": "Smith et al",
    "doi": "10.1234/example"
  },
  "response": "Based on research..."
}
```

This means in audits, you can still see:
- What papers were used for past queries
- What answers were given
- User query history

But you **cannot** retrieve the actual chunks anymore (they're deleted).

---

## Testing Checklist

Before deploying to production:

- [ ] Test deleting a single paper
- [ ] Verify chunks are deleted from `paper_chunks` table
- [ ] Verify embeddings are removed
- [ ] Verify PDF is deleted from GCP
- [ ] Verify query logs remain intact
- [ ] Test deletion of non-existent paper (should raise ValueError)
- [ ] Test GCP deletion failure scenario
- [ ] Test bulk deletion
- [ ] Test rollback on error
- [ ] Test with `delete_from_gcp=False`

---

## Future Enhancements

Possible improvements you could add later:

1. **Soft Delete**: Add `is_deleted` flag instead of hard delete
2. **Deletion Audit Log**: Track who deleted what and when
3. **Scheduled Cleanup**: Cron job to clean up orphaned GCP files
4. **Restore Functionality**: Implement paper restoration from backups
5. **Batch Delete by Criteria**: Delete papers by date, journal, etc.
6. **Export Before Delete**: Automatically export paper data before deletion

---

## Security Considerations

⚠️ **Important:**

1. **Authorization**: This code doesn't include auth checks. Add them before exposing to users:
   ```python
   if not user.is_admin:
       raise PermissionError("Only admins can delete papers")
   ```

2. **Confirmation**: Always require user confirmation before deletion

3. **Audit Trail**: Log all deletions with timestamp and user:
   ```python
   logger.info(f"User {user_id} deleted paper {paper_id} at {timestamp}")
   ```

4. **Backups**: Ensure regular database backups before enabling deletion

5. **Rate Limiting**: Prevent bulk deletion abuse

---

## Troubleshooting

### "Paper not found" error
- Check if paper_id is correct UUID format
- Verify paper exists: `python test_paper_deletion.py list`

### GCP deletion fails
- Check GCP credentials are configured
- Verify bucket name in `settings.gcp_bucket_name`
- Ensure service account has delete permissions
- Check if file exists: `gsutil ls gs://bucket/path`

### Database cascade not working
- Verify foreign key constraint in database
- Check migration `001_initial_schema.py` was applied
- Confirm `CASCADE` is set on `paper_chunks.paper_id`

### Query logs breaking after deletion
- This should NOT happen (no FK constraint)
- If errors occur, check for manual foreign keys added

---

## Support

For issues or questions:
1. Check this documentation first
2. Review `test_paper_deletion.py` for examples
3. Check logs for error details
4. Test with CLI script before using in app

---

## Summary

You now have a **complete paper deletion system** that:

✅ Deletes papers from PostgreSQL
✅ Removes chunks and embeddings automatically
✅ Cleans up PDFs from GCP Storage
✅ Preserves query logs for audit
✅ Handles errors gracefully
✅ Provides detailed results
✅ Supports bulk operations
✅ Ready for Streamlit integration

**Next Steps:**
1. Test with `python test_paper_deletion.py interactive`
2. Verify deletions work as expected
3. Later: Integrate into your Streamlit admin panel
4. Later: Add user data deletion (separate feature)

**Good to go!** 🚀
