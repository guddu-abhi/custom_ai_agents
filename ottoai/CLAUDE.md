# ottoai/ — two-model, code-orchestrated RAG agent

A single user turn is a fixed pipeline of two LLM calls orchestrated by Python
(not by an LLM, not via handoffs), with retrieval as deterministic code between
them: **plan → retrieve → answer**.

- **Planner** (`planner.py`) — SDK `Agent` with `output_type=SearchPlan`
  (`OTTO_PLANNER_MODEL`, default `gpt-4o-mini`). Rewrites the question into a
  search query and extracts structured `ProductFilters`.
- **Retrieval** — deterministic code in `otto_agent.py:_retrieve`. Reuses
  `retrieval.SearchService` with the planner's filters. Over-fetches
  `OTTO_RETRIEVE_K`, narrows to `OTTO_FINAL_K`.
- **Answerer** (`answerer.py`) — SDK `Agent`, no tools, no `output_type`
  (`OTTO_ANSWERER_MODEL`, default `gpt-4o`). `instructions` is generation's
  `SYSTEM_PROMPT`; the run input is `PromptBuilder.build(query, results).user`.

`OttoAgent.stream(query)` yields answer-text deltas, then a final `OttoAnswer`
(answer + citations). `OttoAgent.run(query)` consumes `stream` and returns only
that final `OttoAnswer`.

## Conventions
- Orchestration is code; the flow is fixed. Reuse `retrieval/` + `generation/`
  primitives — `PromptBuilder` and `extract_citations`.
- The answer LLM call goes through the SDK `Runner`, **not**
  `generation.GenerationService`.
- Config via `OTTO_`-prefixed Pydantic settings (`config.py`).
- CLI in `cli.py`; service/agent code stays CLI/HTTP-agnostic.

## Don't
- Don't expose retrieval as a `function_tool` — it runs in code between the two
  model calls (no extra agent-loop turn).
- Don't add handoffs / a manager agent / agents-as-tools on this path.
- Don't add conversation memory this iteration (single-turn pipeline).
- Don't give the Answerer an `output_type` — it blocks token streaming.
- Don't reach for `OPENAI_API_KEY` directly — the SDK reads it from env.
