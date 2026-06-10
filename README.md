# customer-ops-agents

> An Amazon-Rufus–style customer support assistant. Ingests a product catalog,
> retrieves with pgvector, answers with citations, and routes follow-ups through
> specialist agents on the OpenAI Agents SDK.

Built end-to-end so you can run it locally with `docker compose` and Ollama —
no OpenAI key required for the RAG path.

---

## Features

- **Grounded RAG answers** with inline `[pid:123]` citations and per-answer
  grounding metrics (coverage, faithfulness).
- **Pluggable LLMs** — switch between OpenAI and a local Ollama model with a
  flag (`provider=openai|ollama`). The whole RAG path can run offline.
- **Resumable ETL** over the Amazon `meta_Electronics.jsonl` dataset — kill it
  mid-load and rerun from the last checkpoint.
- **pgvector + HNSW cosine search** with strict embedding-space hygiene (query
  and ingest are forced to use the same model).
- **Multi-agent routing** — a triage agent hands off to `product_advisor`,
  `billing`, `account`, or `post_order_support` specialists.
- **FastAPI + SSE chat** at `/chat/conversation` with per-session agent memory.
- **LLM-as-judge eval** built in (`just rag-eval ... judge=true`).

---

## TL;DR — try it in 5 commands

```bash
docker compose up -d                              # Postgres + Adminer
just ollama-pull && just ollama-pull-llm          # one-time model downloads
just alembic                                      # apply schema
just load max-records=500                         # small slice for a fast first run
just generate-local "wireless headphones under \$50"
```

You should see something like:

```
Q: wireless headphones under $50

A: For wireless headphones under $50, the Anker Soundcore Life Q20 [pid:418]
   offers active noise cancelling at ~$45, and the JBL Tune 510BT [pid:1207]
   is a lighter on-ear option around $39. If you need true wireless earbuds,
   the TOZO T6 [pid:2031] sits just under $30.

Citations: [pid:418], [pid:1207], [pid:2031]
Provider:  ollama (qwen2.5:3b-instruct)
Grounding: coverage=1.00  faithfulness=0.93
```

---

## Architecture

```
meta_Electronics.jsonl
        │  (just load)
        ▼
   loader/  ──►  catalog.products + catalog.product_embeddings (pgvector, HNSW)
                                       ▲
                                       │  same Ollama model on both sides
                                       │
                               retrieval/  SearchService
                                       │   (cosine via pgvector <=>)
                                       ▼
                              generation/  GenerationService
                                       │   (OpenAI | Ollama LLM, citations)
                                       ▼
                                tools/   (async @function_tool wrappers)
                                       ▲
                                       │
                              ops_agents/  (triage → product / billing / …)
                                       ▲
                                       │
                                webapp/  FastAPI + SSE  /chat/conversation
```

Layering is enforced by package responsibility:

- `domain/` is the bottom — pure types, imported by everyone, imports nothing
  project-local.
- `loader/` is the **only writer** to `catalog.*`.
- `retrieval/` is **read-only** and reuses the loader's embedder, so query-time
  and ingest-time vectors live in the same space.
- `generation/` opens **no DB connections** — it's a pure
  `(query, results) → grounded answer` function.
- `ops_agents/` orchestrates; data access goes through `tools/`.
- `webapp/` is a thin HTTP/SSE adapter — no SQL, no prompts.

---

## Repo layout

```
.
├── loader/            # ETL: jsonl → catalog.products → embeddings
├── retrieval/         # SearchService + heuristic EvaluationService (CLI)
├── generation/        # GenerationService + LLM providers + grounding eval (CLI)
├── tools/             # @function_tool wrappers exposed to agents
├── ops_agents/        # Triage + specialist agents (Agents SDK)
├── webapp/            # FastAPI app, SSE /chat/conversation
├── domain/            # Pydantic models / frozen dataclasses (bottom layer)
├── db_migrations/     # Alembic, raw SQL only
├── arch_plan/         # ADRs (retrieval, generation layer designs)
├── frontend/          # Static index.html chat client
├── compose.yaml       # postgres (pgvector) + adminer
└── justfile           # canonical entrypoints
```

---

## Stack

| Layer       | Choice                                                            |
|-------------|-------------------------------------------------------------------|
| Runtime     | Python ≥ 3.9, managed with `uv`                                   |
| Database    | PostgreSQL 16 + `pgvector` (cosine, HNSW)                         |
| Embeddings  | Ollama `nomic-embed-text` (768 dims)                              |
| Generation  | OpenAI (default) or local Ollama (e.g. `qwen2.5:3b-instruct`)     |
| Agents      | OpenAI Agents SDK with per-session `SQLAlchemySession` memory     |
| HTTP        | FastAPI + Server-Sent Events                                      |
| Migrations  | Alembic, raw SQL via `op.execute(...)`                            |

---

## Usage

All entrypoints go through `just` (which exports `PYTHONPATH=.` and calls
`uv run` for you).

### Load the catalog

```bash
just load                                       # full ingest, resumable
just load-dry                                   # skip embeddings — quick smoke test
just load max-records=500 batch-size=100        # small slice
just load-reset                                 # clear checkpoint and re-ingest
```

### Search (retrieval only)

```bash
just search "noise cancelling headphones"
just search "usb-c hub for macbook" k=5
just eval   "wireless gaming earbuds" threshold=0.7   # adds heuristic relevance scoring
```

### Generate (RAG end-to-end)

```bash
just generate "wireless headphones under \$50"                       # OpenAI default
just generate-local "best gaming earbuds"                            # offline path
just generate "..." provider=ollama model=qwen2.5:3b-instruct k=5 temperature=0.4
just rag-eval "wireless headphones under \$50"                       # + grounding metrics
just rag-eval "..." judge=true                                       # + LLM-as-judge
```

### Run the chat API

```bash
uv run uvicorn webapp.main:app --reload
# then open frontend/index.html or POST to /chat/conversation
```

### Tests

```bash
just test                                                # loader/tests (default)
uv run pytest retrieval/tests
uv run pytest generation/tests
uv run pytest path/to/test_file.py::TestClass::test_case
```

Tests that hit real OpenAI / Ollama are marked `@pytest.mark.slow` and gated by
env vars (`OPENAI_API_KEY`, `RUN_OLLAMA_TESTS`).

---

## How a conversation flows

1. Client opens an SSE stream to `POST /chat/conversation` with `user_id`,
   `session_id`, and a message.
2. The webapp resolves which agent owns the session (`USER_AGENT_STATE` →
   `ops_agents.registry.get_agent_by_name`), defaulting to `customer_desk_agent`.
3. The agent runs through the OpenAI Agents SDK, calling `tools/` functions
   (which internally use `retrieval.SearchService` + `generation.GenerationService`
   for grounded product answers).
4. The response is an `AgentsBaseResponse` carrying `agent_handed_off_to`; the
   webapp updates the session pointer so the next turn goes to the right specialist.
5. Conversation memory is persisted via `SQLAlchemySession`, scoped per
   `(user_id, session_id)`.

---

## Services and ports

| Service   | Port  | Notes                                                    |
|-----------|-------|----------------------------------------------------------|
| Postgres  | 5432  | `pgvector/pgvector:pg16`; password `example` (dev only)  |
| Adminer   | 8080  | Browser DB UI                                            |
| Ollama    | 11434 | Host-installed; `just ollama-check` verifies             |

---

## Configuration

- `.env` + per-package Pydantic settings, prefixed `LOADER_*`, `RETRIEVAL_*`,
  `GENERATION_*`.
- `GENERATION_PROVIDER=openai|ollama` selects the LLM; CLI `--provider` wins.
- Embeddings are **fixed at 768 dims** by the `nomic-embed-text` choice —
  changing it requires a new migration plus a full re-embed.

---

## Status & roadmap

Working today:
- End-to-end RAG (load → search → grounded answer with citations + metrics).
- Triage + specialist agents with handoff, session memory, SSE streaming.

Planned (see `arch_plan/`):
- Hybrid search (BM25 + vector via RRF) in `retrieval/`.
- `POST /search/answer` HTTP endpoint over `GenerationService`.
- Replace the in-memory `USER_AGENT_STATE` session store with Redis/DB.
- Rewrite `tools/product_tools.py` against the real schema (it currently
  references columns that no longer exist).

---

## Contributing

- Each package has a `CLAUDE.md` with conventions and known gotchas — read the
  one for the package you're touching before opening a PR.
- The non-negotiables: psycopg3 only, SQLAlchemy Core `text()` only (no ORM /
  declarative `Table()`), and migrations are a deploy-time step (never run from
  app startup).

## License

MIT — see `pyproject.toml`.
