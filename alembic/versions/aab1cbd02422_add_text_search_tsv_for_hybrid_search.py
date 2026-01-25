"""add_text_search_tsv_for_hybrid_search

Revision ID: aab1cbd02422
Revises: 7022893f7c55
Create Date: 2026-01-23 10:38:31.710352

Adds text_search_tsv column (TSVECTOR) for hybrid search support in PGVectorStore.
This is a computed column that automatically maintains a full-text search index
based on the 'text' column content.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aab1cbd02422'
down_revision: Union[str, Sequence[str], None] = '7022893f7c55'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add text_search_tsv column for hybrid search."""
    # Add text_search_tsv column as GENERATED STORED
    # This automatically maintains a tsvector for full-text search
    op.execute("""
        ALTER TABLE data_paper_chunks
        ADD COLUMN text_search_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
    """)

    # Create GIN index for fast full-text search
    op.execute("""
        CREATE INDEX ix_data_paper_chunks_text_search_tsv
        ON data_paper_chunks
        USING GIN (text_search_tsv)
    """)


def downgrade() -> None:
    """Remove text_search_tsv column."""
    # Drop index first
    op.execute("DROP INDEX IF EXISTS ix_data_paper_chunks_text_search_tsv")

    # Drop column
    op.execute("ALTER TABLE data_paper_chunks DROP COLUMN text_search_tsv")
