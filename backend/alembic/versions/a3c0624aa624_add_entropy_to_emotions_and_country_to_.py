"""add entropy to emotions and country to users

Revision ID: a3c0624aa624
Revises: ea31fedf1b49
Create Date: 2026-03-07 03:37:26.972259

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'a3c0624aa624'
down_revision = 'ea31fedf1b49'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('emotions', sa.Column('entropy', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_column('emotions', 'entropy')
    