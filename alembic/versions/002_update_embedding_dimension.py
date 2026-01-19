"""Update embedding dimension from 1536 to 3072 for text-embedding-3-large

Revision ID: 002_update_embedding_dimension
Revises: 001_initial_schema
Create Date: 2026-01-16 00:00:00.000000

This migration updates the vector embedding dimension in the paper_chunks table
from 1536 (text-embedding-3-small) to 3072 (text-embedding-3-large).

The migration:
1. Renames the old embedding column
2. Creates a new embedding column with 3072 dimensions
3. Drops the old column
4. Note: No index is created on 3072-dim vectors (pgvector index limit is 2000 dims)
   Vector similarity search will still work, just without index acceleration.
   This is fine for development/testing. For production with millions of vectors,
   consider using a separate specialized vector database.
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

    # Note: No index created for 3072-dim vectors
    # PostgreSQL pgvector extension has a 2000-dimension limit for both
    # IVFFlat and HNSW indexes. Vector similarity search will still work
    # efficiently for development/testing (vectors are still searchable,
    # just without index acceleration).


def downgrade() -> None:
    """Revert embedding dimension from 3072 back to 1536."""

    # Drop the index if it exists
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_embedding_ivfflat")

    # Rename new embedding column
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN embedding TO embedding_new")

    # Create old embedding column with 1536 dimensions
    op.execute("ALTER TABLE paper_chunks ADD COLUMN embedding vector(1536)")

    # Drop the new column
    op.execute("ALTER TABLE paper_chunks DROP COLUMN embedding_new")

    # Recreate the old IVFFlat index (1536 dims is safe for IVFFlat)
    op.execute(
        """
        CREATE INDEX ix_paper_chunks_embedding_ivfflat
        ON paper_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )
