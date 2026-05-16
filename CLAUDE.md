# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## customer-ops-agents

RAG pipeline + Amazon Rufus-like customer support AI agent.
Stack: OpenAI Agents SDK, PostgreSQL 16 + pgvector, Ollama (`nomic-embed-text`, 768-dim),
Alembic, FastAPI (WIP).

## Key Commands
```
docker compose up -d                   # start postgres + adminer + ollama deps
just ollama-pull                       # one-time: pull nomic-embed-text (embeddings)
just ollama-pull-llm                   # one-time: pull qwen2.5:3b-instruct (generation/)
just alembic                           # apply migrations (cd db_migrations/ + alembic upgrade head)
just load                              # run ETL loader (resumable from checkpoint)
just load-reset                        # reset checkpoint + reload from scratch
just load-dry                          # load rows but skip embedding
just search "noise cancelling headphones"   # vector search CLI
just eval   "noise cancelling headphones"   # search + heuristic relevance eval
just generate "wireless headphones under $50"               # RAG answer (OpenAI default)
just generate "..." provider=ollama model=qwen2.5:3b-instruct
just rag-eval "..." judge=true                              # answer + grounding metrics (+ optional LLM judge)
just test                              # runs loader/tests only — other packages have own test dirs
uv run pytest path/to/test_file.py::TestClass::test_case   # run a single test directly
uv run pytest generation/tests          # run generation tests
uv run pytest retrieval/tests           # run retrieval tests
```
All Python entrypoints run via `uv run`; `PYTHONPATH=.` is exported by the justfile.

## Architecture (data flow)

```
meta_Electronics.jsonl
        │  (just load)
        ▼
  loader/  ──► catalog.products + catalog.product_embeddings (pgvector, HNSW)
                                  ▲
                                  │ (same Ollama model)
                                  │
                          retrieval/ SearchService  ──►  generation/ GenerationService
                                                                  │   (OpenAI | Ollama LLM)
                                                                  ▼
                                            tools/  (function_tool wrappers, async, db_session)
                                                                  ▲
                                                                  │
                                         ops_agents/ (Agents SDK; triage → specialists)
                                                                  ▲
                                                                  │
                                              webapp/ FastAPI (SSE /chat/conversation)
```

- **`loader/` is the only writer** to `catalog.*`. **`retrieval/` is read-only** and must reuse `loader.core.embedder` + `loader.db.engine` so query-time and ingest-time embedding spaces stay identical (filter SQL on `e.model_name`).
- **`generation/`** turns `list[SearchResult]` into a grounded, cited answer. No DB access. Pluggable provider (OpenAI / Ollama) via `GENERATION_PROVIDER`. Shared types live in `domain/models/generation.py`. Reused as-is by the future ProductAdvisor tool + webapp endpoint.
- **`ops_agents/` does orchestration only.** Tools live in `tools/`. Agents are wired in `ops_agents/registry.py:AGENT_REGISTRY`; handoffs by name go through `get_agent_by_name`. Triage entrypoint: `customer_desk_agent` (defined in `customer_triage_agent.py`).
- **`domain/` is the bottom layer** — pure Pydantic / frozen dataclasses, no I/O, no imports from other project packages.
- **`db_migrations/`** owns schema. Raw SQL via `op.execute(...)`; no declarative `Table()`. Both `upgrade()` and `downgrade()` must be reversible.
- **`arch_plan/`** holds ADRs (e.g. `retrieval_overview.md` describes planned hybrid search / RRF; `generation_layer_design.md` describes the generation layer).

## Per-layer rules

Every package has its own `CLAUDE.md` with conventions and gotchas. **Read the relevant one before editing**:
`loader/CLAUDE.md`, `retrieval/CLAUDE.md`, `generation/CLAUDE.md`, `tools/CLAUDE.md`, `ops_agents/CLAUDE.md`, `webapp/CLAUDE.md`, `domain/CLAUDE.md`, `db_migrations/CLAUDE.md`.

Notable footguns called out there:
- `loader/db/embed_repo.py:upsert_batch` commits internally — known smell, leave alone unless explicitly tasked.
- `tools/product_tools.py` queries a schema that no longer exists (`products.name/category` vs real `catalog.products.title/main_category`) — rewrite against the real schema or delegate to `retrieval.SearchService` + `generation.GenerationService` before wiring into ProductAdvisor.
- `tools/billing_tools.py`, `tools/support_tools.py`, `tools/db.py` are empty placeholders.
- `webapp` `USER_AGENT_STATE` module-global session store is a stopgap.
- `webapp/api/routers/conversation.py` is the only router; the planned `/search/answer` (generation) router is not yet wired.

## DB / services
- PostgreSQL 16 + pgvector — port 5432 (docker compose)
- Adminer — port 8080
- Ollama — port 11434 (`just ollama-check` to verify)

## Conventions
- Python >=3.9, strict mypy (`disallow_untyped_defs=false`), ruff (line-length 100, target py39).
- psycopg3 only (driver). SQLAlchemy **Core** `text()` only — no ORM, no declarative models.
- All credentials/config via `.env` + Pydantic settings (per-package `*_` env prefix: `LOADER_`, `RETRIEVAL_`, `GENERATION_`).
- DB integration tests use `testcontainers` (pinned to 4.12.0).
- `pytest-asyncio` is in `auto` mode with session-scoped loop.
- Slow tests that hit real external services (OpenAI / Ollama) are marked `@pytest.mark.slow` and gated by env vars (e.g. `OPENAI_API_KEY`, `RUN_OLLAMA_TESTS`).

## Do Not
- Do not use psycopg2.
- Do not introduce SQLAlchemy ORM / declarative `Table()`.
- Do not hardcode credentials or connection strings.
- Do not use `executemany` with `RETURNING` (psycopg3 limitation — row-by-row `execute`, see `ProductRepository.bulk_insert`).
- Do not change the 768-dim embedding without a coordinated migration + re-embed of the corpus.
- Do not run alembic from app startup — migrations are a deploy-time step (`just alembic`).
