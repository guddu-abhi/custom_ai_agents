import json
from collections.abc import AsyncIterator

from agents import Runner, custom_span, trace
from agents.memory import Session
from domain.models.search import (
    ConversationalSearchPlan,
    OttoAnswer,
    OttoTurnState,
    SearchPlan,
    SearchResult,
    ShownProduct,
)
from generation.core.generator import extract_citations
from otto_lib.config import Env
from otto_lib.db.engine import WebAppDBFactory
from otto_lib.logging import get_logger
from ottoai.answerer import answerer_agent
from ottoai.config import settings
from ottoai.conversation_store import (
    append_turn,
    load_state,
    load_transcript,
    render_transcript,
)
from ottoai.filters import merge_filters
from ottoai.otto_agent import OttoAgent
from ottoai.planner import conversational_planner_agent
from retrieval.db.search_repo import SearchRepository

logger = get_logger()


class ConversationalOttoAgent(OttoAgent):
    """Multi-turn Otto. Same plan -> retrieve -> answer pipeline as `OttoAgent`,
    but stateful per `Session`:

    1. load transcript + carried filters/shown-products,
    2. contextual + agentic plan (resolves follow-ups, merges filters, decides
       whether new retrieval is even needed),
    3. conditional retrieve (new vector search, or re-hydrate already-shown
       products by id),
    4. contextual answer,
    5. persist the turn.
    """

    async def _fetch_shown(self, product_ids: list[int]) -> list[SearchResult]:
        async with WebAppDBFactory.get_db_engine(Env(settings.env)).connect() as conn:
            repo = SearchRepository(conn)
            return await repo.fetch_by_ids(product_ids, self._embedder.model_name)

    def _planner_input(
        self, transcript: str, state: OttoTurnState, message: str
    ) -> str:
        active = state.filters.model_dump(exclude_none=True)
        shown = (
            "\n".join(f"[pid:{p.product_id}] {p.title or ''}" for p in state.shown_products)
            or "(none)"
        )
        parts: list[str] = []
        if transcript:
            parts.append(f"CONVERSATION SO FAR:\n{transcript}")
        parts.append(f"CURRENTLY ACTIVE FILTERS: {json.dumps(active) if active else '(none)'}")
        parts.append(f"PRODUCTS CURRENTLY SHOWN:\n{shown}")
        parts.append(f"LATEST USER MESSAGE: {message}")
        return "\n\n".join(parts)

    def _answer_input(self, transcript: str, message: str, results: list[SearchResult]) -> str:
        payload = self._prompt.build(message, results)
        if transcript:
            return f"CONVERSATION SO FAR:\n{transcript}\n\n{payload.user}"
        return payload.user

    async def stream(  # type: ignore[override]
        self, session: Session, message: str
    ) -> AsyncIterator[str | OttoAnswer]:
        with trace("otto conv turn", metadata={"message": message[:200]}):
            logger.info("otto-conv | message={!r}", message)

            state = await load_state(session)
            pairs = await load_transcript(session, settings.history_max_turns)
            transcript = render_transcript(pairs)

            # 1. PLAN (contextual + agentic)
            plan_result = await Runner.run(
                conversational_planner_agent, self._planner_input(transcript, state, message)
            )
            plan: ConversationalSearchPlan = plan_result.final_output
            merged = merge_filters(state.filters, plan.filters, plan.reset_filters)
            logger.info(
                "otto-conv | plan: needs_retrieval={} reset={} query={!r} merged_filters={}",
                plan.needs_retrieval,
                plan.reset_filters,
                plan.query,
                merged.model_dump(exclude_none=True),
            )

            # 2. RETRIEVE (conditional). Fall back to retrieval if the planner
            # says "no search" but we have nothing shown to answer from.
            reuse_shown = not plan.needs_retrieval and bool(state.shown_products)
            if reuse_shown:
                with custom_span("rehydrate-shown", data={"n": len(state.shown_products)}):
                    results = await self._fetch_shown(
                        [p.product_id for p in state.shown_products]
                    )
                shown_products = state.shown_products
            else:
                with custom_span(
                    "retrieve",
                    data={"retrieve_k": settings.retrieve_k, "final_k": settings.final_k},
                ):
                    results = await self._retrieve(
                        SearchPlan(query=plan.query or message, filters=merged)
                    )
                shown_products = [
                    ShownProduct(product_id=r.product_id, title=r.title) for r in results
                ]
            logger.info(
                "otto-conv | {} ({} products)",
                "reused shown" if reuse_shown else "retrieved",
                len(results),
            )

            # 3. ANSWER (contextual). QUESTION is the user's actual message; the
            # contextualized plan.query is only used for retrieval above.
            streamed = Runner.run_streamed(
                answerer_agent, self._answer_input(transcript, message, results)
            )
            chunks: list[str] = []
            async for event in streamed.stream_events():
                if (
                    event.type == "raw_response_event"
                    and event.data.type == "response.output_text.delta"
                ):
                    chunks.append(event.data.delta)
                    yield event.data.delta

            answer = "".join(chunks)
            citations = extract_citations(answer, results)
            logger.info(
                "otto-conv | answer: chars={} citations={}",
                len(answer),
                [c.product_id for c in citations],
            )

            # 4. PERSIST
            await append_turn(
                session,
                message,
                answer,
                OttoTurnState(filters=merged, shown_products=shown_products),
            )
            yield OttoAnswer(query=message, answer=answer, citations=citations)

    async def run(self, session: Session, message: str) -> OttoAnswer:  # type: ignore[override]
        final: OttoAnswer | None = None
        async for item in self.stream(session, message):
            if isinstance(item, OttoAnswer):
                final = item
        assert final is not None, "stream did not yield a final OttoAnswer"
        return final
