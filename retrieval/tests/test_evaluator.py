from decimal import Decimal

import pytest

from domain.models.search import SearchResult
from retrieval.core.evaluator import EvaluationService, _jaccard, _term_coverage, _tokenize


def _result(product_id: int, title: str, content: str, similarity: float) -> SearchResult:
    return SearchResult(
        product_id=product_id,
        title=title,
        main_category="Electronics",
        description=None,
        price=Decimal("9.99"),
        average_rating=Decimal("4.0"),
        content=content,
        similarity=similarity,
    )


def test_tokenize_lowercases_and_strips_stopwords():
    assert _tokenize("The Quick Brown Fox") == {"quick", "brown", "fox"}


def test_tokenize_drops_single_char_and_punctuation():
    assert _tokenize("a, b, foo!! bar.") == {"foo", "bar"}


def test_tokenize_handles_empty_and_none_safely():
    assert _tokenize("") == set()


def test_jaccard_disjoint_is_zero():
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_identical_is_one():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_partial():
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_term_coverage_full():
    assert _term_coverage({"a", "b"}, {"a", "b", "c"}) == 1.0


def test_term_coverage_partial():
    assert _term_coverage({"a", "b", "c"}, {"a"}) == pytest.approx(1 / 3)


def test_term_coverage_empty_query_is_zero():
    assert _term_coverage(set(), {"a"}) == 0.0


def test_evaluate_marks_high_sim_with_overlap_relevant():
    svc = EvaluationService(similarity_threshold=0.6)
    results = [
        _result(1, "Wireless headphones", "noise cancelling over-ear headphones", 0.85),
    ]
    report = svc.evaluate("noise cancelling headphones", results)
    assert report.results[0].relevant is True
    assert report.relevance_ratio == 1.0


def test_evaluate_marks_high_sim_no_overlap_irrelevant():
    svc = EvaluationService(similarity_threshold=0.6)
    results = [
        _result(1, "USB cable", "lightning to usb-c charging cable", 0.95),
    ]
    report = svc.evaluate("noise cancelling headphones", results)
    assert report.results[0].relevant is False


def test_evaluate_marks_low_sim_irrelevant_even_with_overlap():
    svc = EvaluationService(similarity_threshold=0.6)
    results = [
        _result(1, "Headphone case", "case for headphones", 0.40),
    ]
    report = svc.evaluate("noise cancelling headphones", results)
    assert report.results[0].relevant is False


def test_evaluate_summary_aggregates():
    svc = EvaluationService(similarity_threshold=0.5)
    results = [
        _result(1, "wireless headphones", "headphones", 0.90),
        _result(2, "usb cable", "cable", 0.80),
        _result(3, "headphone stand", "stand for headphones", 0.30),
    ]
    report = svc.evaluate("headphones", results)
    assert report.k == 3
    assert report.mean_similarity == pytest.approx((0.90 + 0.80 + 0.30) / 3)
    assert report.relevant_count == 1
    assert report.relevance_ratio == pytest.approx(1 / 3)


def test_evaluate_empty_results():
    svc = EvaluationService()
    report = svc.evaluate("anything", [])
    assert report.k == 0
    assert report.mean_similarity == 0.0
    assert report.relevant_count == 0
    assert report.relevance_ratio == 0.0
