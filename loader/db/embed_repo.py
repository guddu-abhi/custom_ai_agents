from sqlalchemy import Connection, text

from loader.core.embedder import EmbeddingService

_EMBED_INSERT_SQL = text(
    """
    INSERT INTO catalog.product_embeddings
        (product_id, model_name, content, embedding)
    VALUES
        (:product_id, :model_name, :content, CAST(:embedding AS vector))
    ON CONFLICT (product_id, model_name)
    DO UPDATE SET
        content   = EXCLUDED.content,
        embedding = EXCLUDED.embedding
    """
)


class EmbeddingRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def upsert_batch(
        self,
        product_ids: list[int],
        contents: list[str],
        embedder: EmbeddingService,
    ) -> None:
        embeddings = embedder.encode_batch(contents)
        rows = [
            {
                "product_id": pid,
                "model_name": embedder.model_name,
                "content":    content,
                "embedding":  str(emb.tolist()),
            }
            for pid, content, emb in zip(product_ids, contents, embeddings)
        ]
        self._conn.execute(_EMBED_INSERT_SQL, rows)
        self._conn.commit()
