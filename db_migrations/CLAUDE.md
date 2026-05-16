# db_migrations/ — Alembic migrations

Schema is owned here. Every change to `catalog.*` ships as a new migration
file under `alembic/versions/`. Current head includes: `products` table,
`vector` extension + `product_embeddings` table, unique `(product_id,
model_name)` constraint, HNSW index on `product_embeddings.embedding`.

## Conventions
- File name: `YYYYMMDD_HHMMSS_<short_slug>.py`.
- Set `revision` (random hex) and `down_revision` (the previous head) explicitly.
- Use `op.execute("...")` with **raw SQL**. No `op.create_table(...)` /
  declarative `Table()` objects — the project intentionally keeps schema in SQL.
- Both `upgrade()` and `downgrade()` must be implemented and reversible.
- pgvector ops: `<=>` cosine, `<->` L2, `<#>` inner product. We use cosine —
  retrieval surfaces `similarity = 1 - distance`, so the HNSW index must be
  built `WITH (... ) USING hnsw (embedding vector_cosine_ops)`.

## Don't
- Don't edit a migration that has already been applied to anyone's DB. Add a
  new migration that reverses or replaces.
- Don't introduce ORM `Table` / declarative models — schema is raw SQL only.
- Don't change the embedding `vector(768)` dimension without coordinating with
  `loader/config.py:embed_model_name` and rebuilding all embeddings.
- Don't drop the HNSW index in `downgrade()` of unrelated migrations — only the
  migration that added it should drop it.
- Don't run migrations inline (no `alembic` calls in app startup); they're a
  deploy-time step run via `just alembic`.
