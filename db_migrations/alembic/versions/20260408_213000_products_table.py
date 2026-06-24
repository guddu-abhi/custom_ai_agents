"""initial schema — products, FTS trigger, product_embeddings

Revision ID: a3f9c1d2e5b6
Revises: 
Create Date: 2026-04-04 00:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3f9c1d2e5b6"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. products
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE catalog.products (
        id              BIGSERIAL PRIMARY KEY,
        main_category   TEXT,
        title           TEXT,
        average_rating  NUMERIC(3,2) NOT NULL DEFAULT 0,
        rating_number   INTEGER      NOT NULL DEFAULT 0,
        features        TEXT[],
        description     TEXT,
        price           NUMERIC(10,2),
        store           TEXT,
        categories      TEXT[],
        details         JSONB        NOT NULL DEFAULT '{}',
        parent_asin     TEXT NOT NULL,
        created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS catalog.products;")
