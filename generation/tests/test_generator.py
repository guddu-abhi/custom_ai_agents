from decimal import Decimal

from domain.models.generation import Usage
from domain.models.search import SearchResult
from generation.core.generator import GenerationService
from generation.core.prompt import PromptBuilder
from otto_lib.llm.base import GenerationParams, LLMResponse


class FakeProvider:
    name = "openai"

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str, params: GenerationParams) -> LLMResponse:
        self.last_system = system
        self.last_user = user
        return LLMResponse(
            text=self._text,
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


def _result(pid: int, title: str) -> SearchResult:
    return SearchResult(
        product_id=pid,
        title=title,
        main_category="Electronics",
        description="desc",
        price=Decimal("9.99"),
        average_rating=Decimal("4.0"),
        content="desc",
        similarity=0.7,
    )


def test_answer_orchestrates_prompt_and_provider() -> None:
    fake = FakeProvider("Try [pid:1] and also [pid:2]. [pid:1] again.")
    svc = GenerationService(
        provider=fake,
        prompt_builder=PromptBuilder(max_context_chars=6000),
        params=GenerationParams(model="m"),
    )
    results = [_result(1, "Alpha"), _result(2, "Beta")]
    out = svc.answer("q", results)

    assert out.query == "q"
    assert out.provider == "openai"
    assert out.model == "m"
    assert out.usage.total_tokens == 30
    assert out.latency_ms >= 0
    cited = [c.product_id for c in out.citations]
    assert cited == [1, 2]
    assert out.citations[0].title == "Alpha"
    assert fake.last_user is not None and "[pid:1]" in fake.last_user


def test_answer_handles_unknown_citations() -> None:
    fake = FakeProvider("Try [pid:999].")
    svc = GenerationService(
        provider=fake,
        prompt_builder=PromptBuilder(),
        params=GenerationParams(model="m"),
    )
    out = svc.answer("q", [_result(1, "A")])
    assert [c.product_id for c in out.citations] == [999]
    assert out.citations[0].title == ""
    assert out.citations[0].similarity == 0.0
