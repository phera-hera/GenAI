"""Update embedding dimension from 1536 to 3072 for text-embedding-3-large

Revision ID: 002_update_embedding_dimension
Revises: 001_initial_schema
Create Date: 2026-01-16 00:00:00.000000

This migration updates the vector embedding dimension in the paper_chunks table
from 1536 (text-embedding-3-small) to 3072 (text-embedding-3-large).

The migration:
1. Renames the old embedding column
2. Creates a new embedding column with the correct dimension
3. Migrates data (will fail/truncate if vectors don't match, which is expected on first run)
4. Drops the old column
5. Updates the vector index
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_update_embedding_dimension"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[Sequence[str], None] = None
depends_on: Union[Sequence[str], None] = None


def upgrade() -> None:
    """Update embedding dimension from 1536 to 3072."""

    # Drop the old IVFFlat index if it exists
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_embedding_ivfflat")

    # Rename old embedding column
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN embedding TO embedding_old")

    # Create new embedding column with 3072 dimensions
    op.execute("ALTER TABLE paper_chunks ADD COLUMN embedding vector(3072)")

    # Note: We intentionally don't migrate data because:
    # 1. Old 1536-dim vectors can't be directly converted to 3072-dim
    # 2. New papers will generate 3072-dim embeddings
    # 3. On first real data load, old column will be empty anyway

    # Drop the old column
    op.execute("ALTER TABLE paper_chunks DROP COLUMN embedding_old")

    # Recreate the IVFFlat index for 3072-dimensional vectors
    op.execute(
        """
        CREATE INDEX ix_paper_chunks_embedding_ivfflat
        ON paper_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    """Revert embedding dimension from 3072 back to 1536."""

    # Drop the IVFFlat index
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_embedding_ivfflat")

    # Rename new embedding column
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN embedding TO embedding_new")

    # Create old embedding column with 1536 dimensions
    op.execute("ALTER TABLE paper_chunks ADD COLUMN embedding vector(1536)")

    # Drop the new column
    op.execute("ALTER TABLE paper_chunks DROP COLUMN embedding_new")

    # Recreate the old IVFFlat index
    op.execute(
        """
        CREATE INDEX ix_paper_chunks_embedding_ivfflat
        ON paper_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )
