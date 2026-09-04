"""Add per-profile RAG indexing policy.

Revision ID: 0002_profile_rag_indexing
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_profile_rag_indexing"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "newsletter_profiles",
        sa.Column(
            "rag_indexing_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("newsletter_profiles", "rag_indexing_enabled")
