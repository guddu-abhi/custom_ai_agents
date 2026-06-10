from decimal import Decimal

from generation.core.prompt import CITATION_RE, PromptBuilder
from retrieval.db.search_repo import SearchResult


def _result(pid: int, title: str, desc: str = "feature text") -> SearchResult:
    return SearchResult(
        product_id=pid,
        title=title,
        main_category="Electronics",
        description=desc,
        price=Decimal("19.99"),
        average_rating=Decimal("4.50"),
        content=desc,
        similarity=0.85,
    )


def test_citation_regex_matches() -> None:
    text = "Use [pid:42] and [pid:7] but not [pid:abc]."
    pids = [int(m.group(1)) for m in CITATION_RE.finditer(text)]
    assert pids == [42, 7]


def test_build_includes_question_and_pids() -> None:
    pb = PromptBuilder(max_context_chars=6000)
    payload = pb.build("wireless headphones", [_result(1, "A"), _result(2, "B")])
    assert "QUESTION: wireless headphones" in payload.user
    assert "[pid:1]" in payload.user
    assert "[pid:2]" in payload.user
    assert "product recommendation assistant" in payload.system


def test_build_truncates_when_over_budget() -> None:
    pb = PromptBuilder(max_context_chars=200)
    big = _result(1, "Title", desc="x" * 1000)
    payload = pb.build("q", [big])
    assert "…" in payload.user or "truncated" in payload.user.lower()
