import time

from domain.models.generation import Citation, GenerationResult, ProviderName
from generation.core.prompt import CITATION_RE, PromptBuilder
from generation.core.providers.base import GenerationParams, LLMProvider
from retrieval.db.search_repo import SearchResult


class GenerationService:
    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: PromptBuilder,
        params: GenerationParams,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder
        self._params = params

    def answer(self, query: str, results: list[SearchResult]) -> GenerationResult:
        payload = self._prompt_builder.build(query, results)

        start = time.monotonic()
        response = self._provider.complete(payload.system, payload.user, self._params)
        latency_ms = int((time.monotonic() - start) * 1000)

        citations = extract_citations(response.text, results)
        provider_name: ProviderName = self._provider.name  # type: ignore[assignment]
        return GenerationResult(
            query=query,
            answer=response.text,
            citations=citations,
            provider=provider_name,
            model=self._params.model,
            usage=response.usage,
            latency_ms=latency_ms,
        )


def extract_citations(answer: str, results: list[SearchResult]) -> tuple[Citation, ...]:
    by_id = {r.product_id: r for r in results}
    seen: set[int] = set()
    out: list[Citation] = []
    for match in CITATION_RE.finditer(answer):
        pid = int(match.group(1))
        if pid in seen:
            continue
        seen.add(pid)
        r = by_id.get(pid)
        out.append(
            Citation(
                product_id=pid,
                title=(r.title if r and r.title else ""),
                similarity=(r.similarity if r else 0.0),
            )
        )
    return tuple(out)
