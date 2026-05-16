# loader/ — ETL pipeline (write path)

Streams `meta_Electronics.jsonl` → transforms → inserts into `catalog.products` →
batch-embeds via Ollama → upserts into `catalog.product_embeddings`. Resumable
via JSON checkpoint.

## Conventions
- Repository pattern: SQL lives in `db/*_repo.py` only. `core/loader.py` is the
  orchestrator and is the **only** layer that opens transactions.
- SQLAlchemy Core `text(...)` with named bind params. No ORM.
- psycopg3 driver only. Type-hint `conn: sqlalchemy.Connection`.
- Pydantic v2 settings: `model_config = SettingsConfigDict(env_prefix="LOADER_")`.
- Tests in `loader/tests/`. Pure functions get unit tests; DB tests use
  testcontainers (skeleton: `test_product_repo.py`).

## Don't
- Don't add query/read paths here — those live in `retrieval/`.
- Don't `executemany` with `RETURNING` (psycopg3 limitation; use row-by-row
  `execute` like `ProductRepository.bulk_insert`).
- Don't change `embed_model_name` (default `nomic-embed-text`, 768-dim) without
  re-embedding the corpus and updating the pgvector dimension migration.
- Don't introduce SQLAlchemy ORM / declarative models.
- Don't hardcode credentials — `.env` + `LOADER_*` env prefix only.

## Known smell — leave alone unless asked
`db/embed_repo.py:upsert_batch` calls `self._conn.commit()` internally, so the
repo controls a transaction boundary the orchestrator already owns. Untangling
this risks subtly altering load behavior; do not "fix" without an explicit task.
