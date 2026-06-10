# retrieval/ — query path (vector search + heuristic eval)

Embeds a user query with the **same** Ollama model the loader used, runs cosine
similarity against `catalog.product_embeddings` (HNSW-indexed), optionally
scores results with a heuristic evaluator. CLI today; FastAPI endpoint and
agent tool wrap the same `SearchService` later.

## Conventions
- Reuse `loader.core.embedder.EmbeddingService` and
  `loader.db.engine.get_connection`. Do **not** re-implement either.
- Filter SQL on `e.model_name = :model_name` to guard against mixed embedding
  spaces.
- Cosine via pgvector `<=>`. Surface `similarity = 1 - distance` to callers.
- Frozen dataclasses for return types (`SearchResult`, `EvalReport`,
  `ResultEvaluation`) — no leaking ORM rows or dicts.
- CLI lives in `retrieval/cli.py`. Validate at boundaries (e.g.
  `click.IntRange(min=1)` for `--k`). Service classes stay agent/HTTP/CLI-agnostic.

## Don't
- Don't load a different embedding model than the loader. If you need to
  experiment, write a new alembic migration that adds rows under a new
  `model_name`, then pass it through the embedder.
- Don't add LLM calls inside `EvaluationService` — heuristic-only by design.
  LLM-as-judge for retrieval+generation lives in `generation/core/evaluator.py`
  behind the `just rag-eval ... judge=true` flag; don't duplicate it here.
- Don't put HTTP / agent / CLI logic inside `core/`.
- Don't bypass the HNSW index by writing queries that can't use
  `ORDER BY embedding <=> ...` (e.g. wrapping the column in a function).
- Don't add hybrid search / RRF inline — it's a planned next step with its own
  service class. See `arch_plan/retrieval_overview.md`.
