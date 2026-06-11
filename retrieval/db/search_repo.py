from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import Connection, text

from domain.models.search_plan import ProductFilters

_SEARCH_SQL_TMPL = """
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
    {filter_clauses}
    ORDER BY e.embedding <=> CAST(:query_emb AS vector)
    LIMIT :k
    """

_FETCH_BY_IDS_SQL = """
    SELECT
        p.id              AS product_id,
        p.title           AS title,
        p.main_category   AS main_category,
        p.description     AS description,
        p.price           AS price,
        p.average_rating  AS average_rating,
        e.content         AS content
    FROM catalog.products p
    JOIN catalog.product_embeddings e ON e.product_id = p.id
    WHERE p.id = ANY(:ids) AND e.model_name = :model_name
    """


def build_filter_sql(filters: ProductFilters | None) -> tuple[str, dict[str, Any]]:
    if filters is None:
        return "", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}

    if filters.price_max is not None:
        # NULL-safe: ~57% of the catalog has no price. A bare `price <= :max`
        # uses SQL 3-valued logic (NULL <= x is NULL, not true) and would
        # silently drop every unpriced product before ranking. Keep them.
        clauses.append("AND (p.price IS NULL OR p.price <= :price_max)")
        params["price_max"] = filters.price_max
    if filters.min_rating is not None:
        clauses.append("AND p.average_rating >= :min_rating")
        params["min_rating"] = filters.min_rating
    if filters.min_reviews is not None:
        # rating_number is 100%-covered and NOT NULL DEFAULT 0, so this is safe.
        # Pairs with min_rating to avoid surfacing 5.0-from-2-reviews products.
        clauses.append("AND p.rating_number >= :min_reviews")
        params["min_reviews"] = filters.min_reviews
    if filters.brand is not None:
        clauses.append("AND p.store ILIKE :brand")
        params["brand"] = f"%{filters.brand}%"

    return "\n".join(clauses), params


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
        filters: ProductFilters | None = None,
    ) -> list[SearchResult]:
        clause, fparams = build_filter_sql(filters)
        sql = text(_SEARCH_SQL_TMPL.format(filter_clauses=clause))
        rows = self._conn.execute(
            sql,
            {
                "query_emb": str(embedding.tolist()),
                "k": k,
                "model_name": model_name,
                **fparams,
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

    def fetch_by_ids(self, product_ids: list[int], model_name: str) -> list[SearchResult]:
        """Re-hydrate already-shown products by id (no vector search). Used by the
        conversational follow-up path where the user asks about products that were
        surfaced on a previous turn. Returns results in the given id order;
        `similarity` is not meaningful here and is set to 0.0."""
        if not product_ids:
            return []
        rows = (
            self._conn.execute(
                text(_FETCH_BY_IDS_SQL),
                {"ids": product_ids, "model_name": model_name},
            )
            .mappings()
            .all()
        )
        by_id = {
            row["product_id"]: SearchResult(
                product_id=row["product_id"],
                title=row["title"],
                main_category=row["main_category"],
                description=row["description"],
                price=row["price"],
                average_rating=row["average_rating"],
                content=row["content"],
                similarity=0.0,
            )
            for row in rows
        }
        # Preserve the order in which products were originally shown.
        return [by_id[pid] for pid in product_ids if pid in by_id]
