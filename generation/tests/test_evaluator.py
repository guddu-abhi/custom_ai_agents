from decimal import Decimal

from domain.models.generation import GenerationResult, Usage
from domain.models.search import SearchResult
from generation.core.evaluator import EvaluationService


def _result(pid: int, title: str = "T") -> SearchResult:
    return SearchResult(
        product_id=pid,
        title=title,
        main_category="Electronics",
        description="d",
        price=Decimal("1"),
        average_rating=Decimal("4"),
        content="d",
        similarity=0.8,
    )


def _gen(answer: str) -> GenerationResult:
    return GenerationResult(
        query="q",
        answer=answer,
        citations=tuple(),
        provider="openai",
        model="m",
        usage=Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
        latency_ms=42,
    )


def test_faithfulness_all_cited_retrieved() -> None:
    ev = EvaluationService(judge_enabled=False)
    metrics = ev.evaluate(
        "q",
        [_result(1), _result(2)],
        _gen("Recommend [pid:1] and [pid:2]."),
    )
    assert metrics.faithfulness == 1.0
    assert metrics.cited_unknown == ()


def test_faithfulness_some_unknown() -> None:
    ev = EvaluationService(judge_enabled=False)
    metrics = ev.evaluate(
        "q",
        [_result(1)],
        _gen("Use [pid:1] or [pid:999]."),
    )
    assert metrics.faithfulness == 0.5
    assert metrics.cited_unknown == (999,)


def test_faithfulness_zero_when_no_citations() -> None:
    ev = EvaluationService(judge_enabled=False)
    metrics = ev.evaluate("q", [_result(1)], _gen("No citations here."))
    assert metrics.faithfulness == 0.0
    assert metrics.cited_unknown == ()


def test_citation_coverage_sentence_level() -> None:
    ev = EvaluationService(judge_enabled=False)
    answer = "First sentence [pid:1]. Second sentence with no cite. Third one [pid:2]."
    metrics = ev.evaluate("q", [_result(1), _result(2)], _gen(answer))
    assert 0.6 < metrics.citation_coverage < 0.7


def test_metrics_propagate_latency_and_tokens() -> None:
    ev = EvaluationService(judge_enabled=False)
    metrics = ev.evaluate("q", [_result(1)], _gen("Use [pid:1]."))
    assert metrics.latency_ms == 42
    assert metrics.prompt_tokens == 5
    assert metrics.completion_tokens == 7
    assert metrics.judge_score is None
