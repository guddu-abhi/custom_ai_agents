from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from sqlalchemy import Connection, text

_SEARCH_SQL = text(
    """
    SELECT
        p.id              AS product_id,
        p.title           AS title,
        p.main_category   AS main_category,
        p.description     AS description,
        p.price           AS price,
        p.average_rating  AS average_rating,
        e.content         AS content,
        1 - (e.embedding <=> CAST(:query_emb AS vector)) AS similarity
    FROM catalog.product_embeddings e
    JOIN catalog.products p ON p.id = e.product_id
    WHERE e.model_name = :model_name
    ORDER BY e.embedding <=> CAST(:query_emb AS vector)
    LIMIT :k
    """
)


@dataclass(frozen=True)
class SearchResult:
    product_id: int
    title: str | None
    main_category: str | None
    description: str | None
    price: Decimal | None
    average_rating: Decimal
    content: str
    similarity: float


class SearchRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def search_by_vector(
        self,
        embedding: np.ndarray,
        k: int,
        model_name: str,
    ) -> list[SearchResult]:
        rows = self._conn.execute(
            _SEARCH_SQL,
            {
                "query_emb": str(embedding.tolist()),
                "k": k,
                "model_name": model_name,
            },
        ).mappings().all()
        return [
            SearchResult(
                product_id=row["product_id"],
                title=row["title"],
                main_category=row["main_category"],
                description=row["description"],
                price=row["price"],
                average_rating=row["average_rating"],
                content=row["content"],
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]
