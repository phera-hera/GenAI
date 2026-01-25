"""drop_custom_columns_use_metadata_jsonb

Revision ID: 6b9963140451
Revises: aab1cbd02422
Create Date: 2026-01-23 10:46:02.445762

Drop all custom columns (paper_id, chunk_type, chunk_index, section_title, page_number).
Everything goes into metadata_ JSONB following LlamaIndex pattern.

Final schema (LlamaIndex native):
- id: BIGSERIAL PRIMARY KEY
- node_id: VARCHAR (unique)
- text: TEXT
- metadata_: JSONB
- embedding: VECTOR(3072)
- text_search_tsv: TSVECTOR (generated)
- created_at, updated_at: TIMESTAMP
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6b9963140451'
down_revision: Union[str, Sequence[str], None] = 'aab1cbd02422'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop custom columns - use metadata_ JSONB for everything."""

    # Drop foreign key constraint first
    op.execute("ALTER TABLE data_paper_chunks DROP CONSTRAINT IF EXISTS data_paper_chunks_paper_id_fkey")

    # Drop indexes on custom columns
    op.execute("DROP INDEX IF EXISTS ix_data_paper_chunks_paper_id")
    op.execute("DROP INDEX IF EXISTS ix_data_paper_chunks_chunk_type")
    op.execute("DROP INDEX IF EXISTS ix_data_paper_chunks_paper_chunk_type")

    # Drop custom columns
    op.execute("ALTER TABLE data_paper_chunks DROP COLUMN IF EXISTS paper_id")
    op.execute("ALTER TABLE data_paper_chunks DROP COLUMN IF EXISTS chunk_type")
    op.execute("ALTER TABLE data_paper_chunks DROP COLUMN IF EXISTS chunk_index")
    op.execute("ALTER TABLE data_paper_chunks DROP COLUMN IF EXISTS section_title")
    op.execute("ALTER TABLE data_paper_chunks DROP COLUMN IF EXISTS page_number")


def downgrade() -> None:
    """Restore custom columns (not recommended - use metadata_ instead)."""
    # Add columns back
    op.execute("ALTER TABLE data_paper_chunks ADD COLUMN paper_id UUID")
    op.execute("ALTER TABLE data_paper_chunks ADD COLUMN chunk_type VARCHAR(50) DEFAULT 'other'")
    op.execute("ALTER TABLE data_paper_chunks ADD COLUMN chunk_index INTEGER")
    op.execute("ALTER TABLE data_paper_chunks ADD COLUMN section_title VARCHAR(500)")
    op.execute("ALTER TABLE data_paper_chunks ADD COLUMN page_number INTEGER")

    # Restore foreign key
    op.execute("""
        ALTER TABLE data_paper_chunks
        ADD CONSTRAINT data_paper_chunks_paper_id_fkey
        FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
    """)

    # Restore indexes
    op.execute("CREATE INDEX ix_data_paper_chunks_paper_id ON data_paper_chunks(paper_id)")
    op.execute("CREATE INDEX ix_data_paper_chunks_chunk_type ON data_paper_chunks(chunk_type)")
    op.execute("CREATE INDEX ix_data_paper_chunks_paper_chunk_type ON data_paper_chunks(paper_id, chunk_type)")
