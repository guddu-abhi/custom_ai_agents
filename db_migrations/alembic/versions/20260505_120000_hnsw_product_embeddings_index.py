"""HNSW cosine index on catalog.product_embeddings.embedding

Revision ID: e7b4d2c8a915
Revises: c9e2f1a8b3d7
Create Date: 2026-05-05 12:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "e7b4d2c8a915"
down_revision: Union[str, Sequence[str], None] = "c9e2f1a8b3d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX idx_product_embeddings_hnsw
            ON catalog.product_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS catalog.idx_product_embeddings_hnsw")
