from decimal import Decimal

from domain.models.search import SearchResult
from generation.core.generator import extract_citations


def _result(pid: int, title: str, sim: float) -> SearchResult:
    return SearchResult(
        product_id=pid,
        title=title,
        main_category=None,
        description=None,
        price=None,
        average_rating=Decimal("4.0"),
        content="c",
        similarity=sim,
    )


def test_maps_known_pid_to_title_and_similarity():
    results = [_result(1, "Alpha", 0.9), _result(2, "Beta", 0.8)]
    cites = extract_citations("buy [pid:1]", results)
    assert len(cites) == 1
    assert cites[0].product_id == 1
    assert cites[0].title == "Alpha"
    assert cites[0].similarity == 0.9


def test_unknown_pid_is_not_dropped():
    # The reused extract_citations keeps cited pids absent from results, with
    # an empty title and 0.0 similarity (grounding is measured elsewhere).
    results = [_result(1, "Alpha", 0.9)]
    cites = extract_citations("see [pid:1] and [pid:99]", results)
    by_id = {c.product_id: c for c in cites}
    assert set(by_id) == {1, 99}
    assert by_id[99].title == ""
    assert by_id[99].similarity == 0.0


def test_repeated_pid_is_deduped():
    results = [_result(1, "Alpha", 0.9)]
    cites = extract_citations("[pid:1] ... [pid:1]", results)
    assert len(cites) == 1
