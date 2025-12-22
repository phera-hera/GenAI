"""Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2024-01-01 00:00:00.000000

Creates the initial database schema for FemTech Medical RAG Agent:
- users: Basic user information
- health_profiles: User health and symptom data
- papers: Medical research paper metadata
- paper_chunks: Chunked paper content with vector embeddings
- query_logs: Query history for compliance and improvement

Also creates:
- pgvector extension (if not exists)
- uuid-ossp extension (if not exists)
- IVFFlat index for vector similarity search
- Various performance indexes
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables and indexes."""
    
    # Ensure extensions are enabled (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # =========================================================================
    # Users Table
    # =========================================================================
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column(
            "external_id",
            sa.String(length=255),
            nullable=True,
            comment="External auth provider ID (e.g., Zitadel)",
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    # =========================================================================
    # Health Profiles Table
    # =========================================================================
    op.create_table(
        "health_profiles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("ethnicity", sa.String(length=100), nullable=True),
        sa.Column(
            "symptoms",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Current symptoms as structured data",
        ),
        sa.Column(
            "medical_history",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Relevant medical history",
        ),
        sa.Column(
            "additional_info",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional health context",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index(
        op.f("ix_health_profiles_user_id"), "health_profiles", ["user_id"], unique=True
    )

    # =========================================================================
    # Papers Table
    # =========================================================================
    op.create_table(
        "papers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "authors",
            sa.Text(),
            nullable=True,
            comment="Comma-separated author names",
        ),
        sa.Column("journal", sa.String(length=500), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("doi", sa.String(length=255), nullable=True),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column(
            "gcp_path",
            sa.String(length=1000),
            nullable=False,
            comment="Path to PDF in GCP Cloud Storage",
        ),
        sa.Column(
            "file_hash",
            sa.String(length=64),
            nullable=True,
            comment="SHA-256 hash for deduplication",
        ),
        sa.Column(
            "is_processed", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("processing_error", sa.Text(), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "extra_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional paper metadata",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doi"),
        sa.UniqueConstraint("file_hash"),
        sa.UniqueConstraint("gcp_path"),
    )
    op.create_index(op.f("ix_papers_doi"), "papers", ["doi"], unique=True)
    op.create_index(
        op.f("ix_papers_publication_year"), "papers", ["publication_year"], unique=False
    )
    op.create_index(
        op.f("ix_papers_is_processed"), "papers", ["is_processed"], unique=False
    )

    # =========================================================================
    # Paper Chunks Table (with vector embeddings)
    # Using raw SQL for the vector column since Alembic doesn't natively support it
    # =========================================================================
    op.execute(
        """
        CREATE TABLE paper_chunks (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            chunk_type VARCHAR(50) NOT NULL DEFAULT 'other',
            chunk_index INTEGER NOT NULL,
            embedding vector(1536),
            section_title VARCHAR(500),
            page_number INTEGER,
            chunk_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """
    )
    
    # Add comments
    op.execute(
        "COMMENT ON COLUMN paper_chunks.chunk_index IS 'Order of chunk within the paper'"
    )
    op.execute(
        "COMMENT ON COLUMN paper_chunks.chunk_metadata IS 'Additional chunk metadata (e.g., table structure)'"
    )
    
    op.create_index(
        op.f("ix_paper_chunks_paper_id"), "paper_chunks", ["paper_id"], unique=False
    )
    op.create_index(
        op.f("ix_paper_chunks_chunk_type"), "paper_chunks", ["chunk_type"], unique=False
    )
    op.create_index(
        "ix_paper_chunks_paper_chunk_type",
        "paper_chunks",
        ["paper_id", "chunk_type"],
        unique=False,
    )

    # =========================================================================
    # Query Logs Table
    # =========================================================================
    op.create_table(
        "query_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("ph_value", sa.Float(), nullable=False),
        sa.Column(
            "query_text",
            sa.Text(),
            nullable=True,
            comment="Additional user query text if provided",
        ),
        sa.Column(
            "health_profile_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Snapshot of health profile at query time",
        ),
        sa.Column(
            "risk_level",
            sa.String(length=20),
            nullable=False,
            server_default="normal",
        ),
        sa.Column(
            "retrieved_chunk_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="IDs of chunks retrieved for this query",
        ),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Paper citations used in response",
        ),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "feedback_rating",
            sa.Integer(),
            nullable=True,
            comment="User rating 1-5",
        ),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_query_logs_user_id"), "query_logs", ["user_id"], unique=False
    )
    op.create_index(
        op.f("ix_query_logs_risk_level"), "query_logs", ["risk_level"], unique=False
    )
    op.create_index(
        "ix_query_logs_user_created",
        "query_logs",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_query_logs_risk_level_created",
        "query_logs",
        ["risk_level", "created_at"],
        unique=False,
    )

    # =========================================================================
    # Create IVFFlat Index for Vector Similarity Search
    # =========================================================================
    # Note: IVFFlat requires some data to build the index effectively.
    # For now, we create it with a small number of lists.
    # In production, you may want to rebuild with more lists after loading data.
    op.execute(
        """
        CREATE INDEX ix_paper_chunks_embedding_ivfflat 
        ON paper_chunks 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
        """
    )

    # =========================================================================
    # Create trigger function for updated_at
    # =========================================================================
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
        """
    )

    # Create triggers for each table
    for table in ["users", "health_profiles", "papers", "paper_chunks", "query_logs"]:
        op.execute(
            f"""
            CREATE TRIGGER update_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column()
            """
        )


def downgrade() -> None:
    """Drop all tables and indexes."""
    
    # Drop triggers
    for table in ["users", "health_profiles", "papers", "paper_chunks", "query_logs"]:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")
    
    # Drop trigger function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    # Drop IVFFlat index
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_embedding_ivfflat")
    
    # Drop tables in reverse order of creation (respecting foreign keys)
    op.drop_table("query_logs")
    op.drop_table("paper_chunks")
    op.drop_table("papers")
    op.drop_table("health_profiles")
    op.drop_table("users")
    
    # Note: We don't drop the extensions as they may be used by other applications
