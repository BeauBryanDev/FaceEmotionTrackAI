"""add_is_superuser_to_users

Revision ID: 8f225089f7e7
Revises: bb9779230ae7
Create Date: 2026-02-25 06:08:04.806918

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8f225089f7e7'
down_revision = 'bb9779230ae7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column(
            'is_superuser',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false')
        )
    )


def downgrade() -> None:
    op.drop_column('users', 'is_superuser')
