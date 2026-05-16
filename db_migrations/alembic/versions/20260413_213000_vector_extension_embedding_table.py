"""vector extension, embed_tier enum, and product_embeddings table

Revision ID: b4d7c1e9f203
Revises: a3f9c1d2e5b6
Create Date: 2026-04-13 21:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b4d7c1e9f203"
down_revision: Union[str, Sequence[str], None] = "a3f9c1d2e5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE catalog.product_embeddings (
            id          BIGSERIAL PRIMARY KEY,
            product_id  BIGINT              NOT NULL REFERENCES catalog.products(id) ON DELETE CASCADE,
            model_name  TEXT                NOT NULL,
            content     TEXT                NOT NULL,
            embedding   vector(768),
            created_at  TIMESTAMPTZ         NOT NULL DEFAULT now()
        )
        """
    )



def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS catalog.product_embeddings")
