# generation/ — RAG answer synthesis (no DB, no I/O beyond LLM)

Takes `list[SearchResult]` from `retrieval/` and turns it into a grounded,
citation-bearing answer. Pluggable provider (OpenAI or Ollama). Sync
`GenerationService.answer(query, results) -> GenerationResult`. Reused as-is by
the future ProductAdvisor tool and the webapp endpoint.

## Conventions
- Shared types live in `domain/models/generation.py` so `tools/`, `webapp/`,
  `ops_agents/` import them without depending on `generation/`.
- Citation token is `[pid:<int>]` (regex in `core/prompt.py:CITATION_RE`).
  `evaluator.py` imports the regex from there — do not redefine it.
- Provider implementations live in `core/providers/`. Use the `LLMProvider`
  Protocol; choose by name via `get_provider(...)`. Providers receive
  already-resolved values from `config.py`. **Do not read env vars inside a
  provider.**
- Reuse `retrieval.SearchService` + `loader.core.embedder.EmbeddingService`.
  `generation/` itself opens **no DB connections**.
- CLI lives in `generation/cli.py`. Validate at boundaries (e.g.
  `click.IntRange(min=1)`). Service classes stay agent/HTTP/CLI-agnostic.

## Don't
- Don't add streaming / SSE — service is sync.
- Don't add reranking, hybrid search, or query rewriting here — those belong
  in `retrieval/`.
- Don't add conversation memory / multi-turn state — that lives in
  `ops_agents/` + `domain/`.
- Don't add function-calling inside the LLM call. Generation is a single
  prompt → answer step.
- Don't do direct DB access. If a future feature needs persistence, that is a
  separate ADR — do not pre-add `db/`.
- Don't reach for `OPENAI_API_KEY` directly — the `openai` SDK reads it itself.
