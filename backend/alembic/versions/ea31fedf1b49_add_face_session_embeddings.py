"""add_face_session_embeddings

Revision ID: ea31fedf1b49
Revises: 8f225089f7e7
Create Date: 2026-02-27 03:52:32.158191

"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision = 'ea31fedf1b49'
down_revision = '8f225089f7e7'
branch_labels = None
depends_on = None

def upgrade() -> None:
    
    op.create_table(
        "face_session_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=512),
            nullable=False
        ),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id")
    )
    op.create_index(
        op.f("ix_face_session_embeddings_id"),
        "face_session_embeddings",
        ["id"],
        unique=False
    )
    op.create_index(
        op.f("ix_face_session_embeddings_user_id"),
        "face_session_embeddings",
        ["user_id"],
        unique=False
    )
    op.create_index(
        op.f("ix_face_session_embeddings_session_id"),
        "face_session_embeddings",
        ["session_id"],
        unique=False
    )
    op.create_index(
        op.f("ix_face_session_embeddings_captured_at"),
        "face_session_embeddings",
        ["captured_at"],
        unique=False
    )
    
    
def downgrade() -> None:
    
    op.drop_index(
        op.f("ix_face_session_embeddings_captured_at"),
        table_name="face_session_embeddings"
    )
    op.drop_index(
        op.f("ix_face_session_embeddings_session_id"),
        table_name="face_session_embeddings"
    )
    op.drop_index(
        op.f("ix_face_session_embeddings_user_id"),
        table_name="face_session_embeddings"
    )
    op.drop_index(
        op.f("ix_face_session_embeddings_id"),
        table_name="face_session_embeddings"
    )
    op.drop_table("face_session_embeddings")
