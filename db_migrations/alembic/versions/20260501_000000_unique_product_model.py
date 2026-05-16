"""Add UNIQUE(product_id, model_name) to catalog.product_embeddings

Revision ID: c9e2f1a8b3d7
Revises: b4d7c1e9f203
Create Date: 2026-05-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "c9e2f1a8b3d7"
down_revision: Union[str, Sequence[str], None] = "b4d7c1e9f203"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE catalog.product_embeddings
        ADD CONSTRAINT uq_product_embeddings_product_model
        UNIQUE (product_id, model_name)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE catalog.product_embeddings
        DROP CONSTRAINT IF EXISTS uq_product_embeddings_product_model
        """
    )
