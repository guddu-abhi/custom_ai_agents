import asyncio
from collections.abc import AsyncIterator

from agents import Runner, custom_span, trace
from loguru import logger

from domain.models.search_plan import OttoAnswer, SearchPlan
from generation.core.generator import extract_citations
from generation.core.prompt import PromptBuilder
from loader.config import settings as loader_settings
from loader.core.embedder import EmbeddingService
from loader.db.engine import get_connection
from ottoai.answerer import answerer_agent
from ottoai.config import settings
from ottoai.planner import planner_agent
from retrieval.core.searcher import SearchService
from retrieval.db.search_repo import SearchRepository, SearchResult


class OttoAgent:
    def __init__(self) -> None:
        self._embedder = EmbeddingService(
            loader_settings.embed_model_name, loader_settings.ollama_base_url
        )
        self._prompt = PromptBuilder(max_context_chars=settings.max_context_chars)

    def _retrieve(self, plan: SearchPlan) -> list[SearchResult]:
        with get_connection(settings.env) as conn:
            searcher = SearchService(self._embedder, SearchRepository(conn))
            results = searcher.search(plan.query, k=settings.retrieve_k, filters=plan.filters)

            # Auto-relax: structured filters (brand / price_max / min_rating) can
            # over-constrain on sparse metadata and return nothing. Rather than
            # leave the answerer empty-handed, retry once on the same vector query
            # with filters dropped so the user still gets relevant products.
            if not results and plan.filters.model_dump(exclude_none=True):
                logger.info(
                    "otto | filtered search returned 0 rows (filters={}); "
                    "relaxing filters and retrying on the vector query",
                    plan.filters.model_dump(exclude_none=True),
                )
                results = searcher.search(plan.query, k=settings.retrieve_k, filters=None)
        return results[: settings.final_k]

    async def stream(self, query: str) -> AsyncIterator[str | OttoAnswer]:
        # One trace per user turn unifies planner + retrieval + answerer into a
        # single timeline at https://platform.openai.com/traces.
        with trace("otto turn", metadata={"query": query[:200]}):
            logger.info("otto | query={!r}", query)

            # 1. PLAN
            plan_result = await Runner.run(planner_agent, query)
            plan: SearchPlan = plan_result.final_output
            logger.info(
                "otto | plan: query={!r} filters={}",
                plan.query,
                plan.filters.model_dump(exclude_none=True),
            )

            # 2. RETRIEVE (deterministic code; custom_span makes it visible in the trace)
            with custom_span(
                "retrieve",
                data={"retrieve_k": settings.retrieve_k, "final_k": settings.final_k},
            ):
                results = await asyncio.to_thread(self._retrieve, plan)
            logger.info(
                "otto | retrieved {} results (top: {})",
                len(results),
                [(r.product_id, round(r.similarity, 3)) for r in results[:5]],
            )

            # 3. ANSWER
            payload = self._prompt.build(plan.query, results)
            streamed = Runner.run_streamed(answerer_agent, payload.user)
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
                "otto | answer: chars={} citations={}",
                len(answer),
                [c.product_id for c in citations],
            )
            if citations:
                in_list = ", ".join(str(c.product_id) for c in citations)
                logger.info(
                    'otto | cited products SQL: select * from "catalog".products p '
                    "where p.id in ({});",
                    in_list,
                )
            yield OttoAnswer(query=query, answer=answer, citations=citations)

    async def run(self, query: str) -> OttoAnswer:
        final: OttoAnswer | None = None
        async for item in self.stream(query):
            if isinstance(item, OttoAnswer):
                final = item
        assert final is not None, "stream did not yield a final OttoAnswer"
        return final
