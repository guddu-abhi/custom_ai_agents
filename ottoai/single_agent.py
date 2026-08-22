"""Agentic single-agent Otto.

Unlike `OttoAgent`'s fixed, code-orchestrated `plan -> retrieve -> answer`
pipeline, this path is a SINGLE `gpt-4o` agent that owns query planning AND
answer generation, and calls the retriever as an SDK `function_tool` whenever it
decides it needs products. The agent loop (not Python) drives the flow: it may
search zero, one, or many times per turn, and across turns it can answer
follow-ups from earlier tool results or re-search.

Memory is the SDK `Session` passed straight to `Runner.run_streamed(session=...)`
(auto-memory: messages + tool calls/results persist), so multi-turn context is
free. The retrieval SQL/filter behavior is reused unchanged via
`ottoai.otto_agent.retrieve_products`.
"""

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field

import mlflow

from agents import Agent, RunContextWrapper, Runner, function_tool, trace
from agents.memory import Session
from domain.models.search import OttoAnswer, ProductFilters, SearchPlan, SearchResult
from generation.core.generator import extract_citations
from generation.core.prompt import SYSTEM_PROMPT, PromptBuilder
from loader.config import settings as loader_settings
from otto_lib.embedding import EmbeddingService
from otto_lib.logging import get_logger
from otto_lib.tracing import tag_session
from ottoai.config import settings
from ottoai.otto_agent import retrieve_products

logger = get_logger()


@dataclass
class OttoToolContext:
    """Per-run context handed to the tool via `RunContextWrapper`.

    Carries the shared embedder + prompt builder, and accumulates every product
    the tool returns this turn (`seen`) so citations can be mapped back after the
    answer stream closes — even across multiple tool calls in one turn.
    """

    embedder: EmbeddingService
    prompt: PromptBuilder
    seen: dict[int, SearchResult] = field(default_factory=dict)


@function_tool
async def search_products(
    ctx: RunContextWrapper[OttoToolContext],
    query: str,
    brand: str | None = None,
    price_max: float | None = None,
    min_rating: float | None = None,
    min_reviews: int | None = None,
) -> str:
    """Semantic product search over the catalog. Call this whenever you need real
    product data to answer the user.

    Args:
        query: A descriptive keyword search query that KEEPS the product
            type/category (e.g. 'noise cancelling headphones', 'usb-c laptop
            charger'). This feeds a semantic vector search, so descriptive
            product words improve results.
        brand: Brand / manufacturer name. Set ONLY if the user named one.
        price_max: Maximum price in USD. Set ONLY if the user gave a ceiling.
        min_rating: Minimum average star rating (0-5). Set ONLY if asked.
        min_reviews: Minimum number of customer reviews. Set when the user wants
            popular / well-reviewed / best-selling products, and ALWAYS set it to
            at least 50 whenever you set `min_rating`.

    Returns:
        A formatted list of candidate products, each tagged `[pid:<id>]`. Cite
        the products you use by their `[pid:<id>]` tokens. Returns a notice if
        nothing matched.
    """
    filters = ProductFilters(
        brand=brand,
        price_max=price_max,
        min_rating=min_rating,
        min_reviews=min_reviews,
    )
    # relax=False: return honest results. Unlike the one-shot OttoAgent pipeline,
    # this agent can re-search, so we must NOT silently swap in filter-dropped
    # (off-target) products — that would mislead the agent and pollute citations.
    results = await retrieve_products(
        ctx.context.embedder,
        SearchPlan(query=query, filters=filters),
        False,
    )
    logger.info(
        "otto-single | tool search: query={!r} filters={} -> {} results",
        query,
        filters.model_dump(exclude_none=True),
        len(results),
    )
    for r in results:
        ctx.context.seen[r.product_id] = r
    if not results:
        active = filters.model_dump(exclude_none=True)
        return (
            f"No products matched query={query!r}"
            + (f" with filters {active}" if active else "")
            + ". The catalog may still contain matching items that this query "
            "missed — retry with a more descriptive query (include the brand "
            "and/or product type, e.g. the brand name itself), or drop an "
            "over-narrow filter. Do NOT tell the user the catalog has none "
            "until a broader retry also comes back empty."
        )
    # Same budgeted, [pid:N]-tagged product block the answerer normally consumes.
    return ctx.context.prompt.build(query, results).user


SINGLE_AGENT_INSTRUCTIONS = (
    "You are Otto, a product recommendation assistant for an electronics catalog. "
    "You plan, search, and answer on your own using the `search_products` tool.\n\n"
    "WHEN TO SEARCH:\n"
    "- Call `search_products` whenever you need real product data to answer. You "
    "may call it multiple times in a turn (e.g. to broaden, narrow, or compare).\n"
    "- For follow-up questions about products you already retrieved earlier in the "
    "conversation (e.g. 'which is cheapest?', 'is the second one waterproof?'), "
    "answer from those prior tool results WITHOUT searching again.\n"
    "- Re-search when the user wants different/new/more products or changes a "
    "constraint (brand, budget, rating, popularity).\n\n"
    "BUILDING THE SEARCH:\n"
    "- `query`: rewrite the user's request into a concise keyword query and KEEP "
    "the product type/category in it (embeddings handle category; there is no "
    "category filter). NEVER use a bare generic word like 'electronics' or "
    "'products' as the query — it carries no semantic signal and finds nothing "
    "useful.\n"
    "- Brand/store browse: if the user just wants to see what a brand/store "
    "carries WITHOUT naming a product type (e.g. 'anything from Sony?'), put the "
    "BRAND NAME itself in `query` (query='Sony') AND set `brand`. Do not invent a "
    "product type they didn't ask for.\n"
    "- Set `brand`, `price_max`, `min_rating`, `min_reviews` ONLY when the user "
    "states them explicitly. Do not invent filters.\n"
    "- RECOVERING FROM EMPTY: if a search returns no products, do NOT conclude the "
    "catalog has none. Retry with a broader / reworded query (e.g. just the brand "
    "name) or with fewer filters first. Only after a broader retry is also empty "
    "should you tell the user nothing was found.\n"
    "- min_reviews: set when the user asks for popular / well-reviewed / trusted / "
    "best-selling products (~100), or 'some reviews' (~50).\n"
    "- Pairing rule: average rating alone is unreliable (a 5.0 from 2 reviews "
    "beats a 4.6 from 10k). Whenever you set `min_rating`, also set `min_reviews` "
    "to at least 50.\n\n"
    "ANSWERING:\n" + SYSTEM_PROMPT
)


class SingleAgentOtto:
    """Agentic, multi-turn Otto: one `gpt-4o` agent with the retriever as a tool.

    `stream(session, message)` yields answer-text deltas, then a final
    `OttoAnswer` (answer + citations). `run(...)` consumes `stream` and returns
    only that final `OttoAnswer`. Mirrors `ConversationalOttoAgent`'s contract so
    the webapp SSE wrapper is identical.
    """

    def __init__(self) -> None:
        self._embedder = EmbeddingService(
            loader_settings.embed_model_name, loader_settings.ollama_base_url
        )
        self._prompt = PromptBuilder(max_context_chars=settings.max_context_chars)
        self._agent = Agent(
            name="OttoSingleAgent",
            instructions=SINGLE_AGENT_INSTRUCTIONS,
            model=settings.single_agent_model,
            tools=[search_products],
        )
# query : find matching products + accessories + compatible lenses
# hybrid search - RRF
    async def stream(
        self,
        session: Session,
        message: str,
        session_id: str,
        user_id: str | None = None,
    ) -> AsyncIterator[str | OttoAnswer]:
        # Own an MLflow span that wraps the whole turn so the trace's Inputs /
        # Outputs carry the FULL request + response JSON (not just the answer
        # string autolog captures). The autolog LLM/tool spans nest underneath
        # this span, and it becomes the trace root.
        with mlflow.start_span(name="otto single-agent turn") as span:
            # Full request JSON -> trace Inputs (mirrors the HTTP request body).
            span.set_inputs(
                {"user_id": user_id, "session_id": session_id, "message": message}
            )
            with trace("otto single-agent turn", metadata={"message": message[:200]}):
                logger.info("otto-single | message={!r}", message)

                # Per-run accumulator the tool writes into; `session` gives the agent
                # its multi-turn memory (messages + tool calls/results) for free.
                ctx = OttoToolContext(embedder=self._embedder, prompt=self._prompt)
                streamed = Runner.run_streamed(
                    self._agent, message, context=ctx, session=session
                )
                chunks: list[str] = []
                tagged = False
                async for event in streamed.stream_events():
                    # Tag this turn's trace with the conversation session on the first
                    # event (the MLflow trace is live once the run starts streaming),
                    # so every turn of a session groups together in the MLflow UI.
                    if not tagged:
                        tag_session(session_id, user_id)
                        tagged = True
                    if (
                        event.type == "raw_response_event"
                        and event.data.type == "response.output_text.delta"
                    ):
                        chunks.append(event.data.delta)
                        yield event.data.delta

                answer = "".join(chunks)
                citations = extract_citations(answer, list(ctx.seen.values()))

            # Full response JSON -> trace Outputs (mirrors the endpoint return).
            span.set_outputs(
                {
                    "answer": answer,
                    "citations": [asdict(c) for c in citations],
                    "session_id": session_id,
                }
            )
            logger.info(
                "otto-single | answer: chars={} tool_products={} citations={}",
                len(answer),
                len(ctx.seen),
                [c.product_id for c in citations],
            )
            yield OttoAnswer(query=message, answer=answer, citations=citations)

    async def run(
        self,
        session: Session,
        message: str,
        session_id: str,
        user_id: str | None = None,
    ) -> OttoAnswer:
        final: OttoAnswer | None = None
        async for item in self.stream(session, message, session_id, user_id):
            if isinstance(item, OttoAnswer):
                final = item
        assert final is not None, "stream did not yield a final OttoAnswer"
        return final
