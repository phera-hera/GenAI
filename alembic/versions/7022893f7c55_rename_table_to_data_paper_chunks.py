"""rename_table_to_data_paper_chunks

Revision ID: 7022893f7c55
Revises: 296c0237b66a
Create Date: 2026-01-23 10:35:13.930946

Renames paper_chunks table to data_paper_chunks to align with LlamaIndex PGVectorStore
table naming convention (prefix: data_).

This allows native PGVectorStore.add() to work without custom wrappers.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7022893f7c55'
down_revision: Union[str, Sequence[str], None] = '296c0237b66a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename paper_chunks to data_paper_chunks."""
    # Rename the table (all constraints, indexes, and sequences are automatically renamed)
    op.rename_table('paper_chunks', 'data_paper_chunks')


def downgrade() -> None:
    """Rename data_paper_chunks back to paper_chunks."""
    # Rename back
    op.rename_table('data_paper_chunks', 'paper_chunks')
