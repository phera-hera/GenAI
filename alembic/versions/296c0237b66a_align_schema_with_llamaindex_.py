"""align_schema_with_llamaindex_pgvectorstore

Revision ID: 296c0237b66a
Revises: 002_update_embedding_dimension
Create Date: 2026-01-23 10:30:34.684916

Aligns paper_chunks schema with LlamaIndex PGVectorStore expectations:
- Renames 'id' → 'node_id' (UUID → VARCHAR)
- Adds new 'id' BIGSERIAL PRIMARY KEY
- Renames 'content' → 'text'
- Renames 'chunk_metadata' → 'metadata_'

This allows us to use native PGVectorStore.add() without custom wrappers.
All our custom columns (paper_id, chunk_type, etc.) are preserved.

BREAKING CHANGE: Requires re-ingesting all papers.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '296c0237b66a'
down_revision: Union[str, Sequence[str], None] = '002_update_embedding_dimension'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Align schema with LlamaIndex PGVectorStore."""

    # Step 1: Drop existing primary key constraint on 'id'
    op.execute("ALTER TABLE paper_chunks DROP CONSTRAINT paper_chunks_pkey")

    # Step 2: Rename 'id' column to 'node_id' and convert UUID to VARCHAR
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN id TO node_id")
    op.execute("ALTER TABLE paper_chunks ALTER COLUMN node_id TYPE VARCHAR USING node_id::VARCHAR")

    # Step 3: Add new 'id' column as BIGSERIAL (auto-incrementing) and make it primary key
    op.execute("ALTER TABLE paper_chunks ADD COLUMN id BIGSERIAL PRIMARY KEY")

    # Step 4: Rename 'content' to 'text'
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN content TO text")

    # Step 5: Rename 'chunk_metadata' to 'metadata_'
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN chunk_metadata TO metadata_")

    # Note: All indexes and foreign keys remain intact since they reference other columns


def downgrade() -> None:
    """Revert to original schema."""

    # Step 1: Rename columns back
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN text TO content")
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN metadata_ TO chunk_metadata")

    # Step 2: Drop the new BIGSERIAL id column
    op.execute("ALTER TABLE paper_chunks DROP COLUMN id")

    # Step 3: Rename node_id back to id and convert VARCHAR to UUID
    op.execute("ALTER TABLE paper_chunks ALTER COLUMN node_id TYPE UUID USING node_id::UUID")
    op.execute("ALTER TABLE paper_chunks RENAME COLUMN node_id TO id")

    # Step 4: Re-add primary key constraint on 'id'
    op.execute("ALTER TABLE paper_chunks ADD PRIMARY KEY (id)")
