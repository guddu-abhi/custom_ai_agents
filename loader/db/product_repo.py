from sqlalchemy import Connection, String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

_INSERT_SQL = text(
    """
    INSERT INTO catalog.products (
        main_category, title, average_rating, rating_number,
        features, description, price, store,
        categories, details, parent_asin
    ) VALUES (
        :main_category, :title, :average_rating, :rating_number,
        :features, :description, :price, :store,
        :categories, :details, :parent_asin
    )
    RETURNING id
    """
).bindparams(
    bindparam("features",   type_=ARRAY(String)),
    bindparam("categories", type_=ARRAY(String)),
    bindparam("details",    type_=JSONB()),
)


class ProductRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def bulk_insert(self, rows: list[dict]) -> list[int]:
        ids = []
        for row in rows:
            result = self._conn.execute(_INSERT_SQL, row)
            ids.append(result.scalar_one())
        return ids
