import re
from dataclasses import dataclass

from domain.models.search import SearchResult

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
        "at", "for", "with", "by", "from", "as", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "this", "that", "these", "those", "it", "its", "i", "me", "my",
        "you", "your", "he", "she", "we", "they", "them", "what", "which",
        "who", "how", "when", "where", "why",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


@dataclass(frozen=True)
class ResultEvaluation:
    product_id: int
    similarity: float
    lexical_overlap: float
    term_coverage: float
    relevant: bool


@dataclass(frozen=True)
class EvalReport:
    query: str
    k: int
    threshold: float
    results: list[ResultEvaluation]
    mean_similarity: float
    relevant_count: int
    relevance_ratio: float


class EvaluationService:
    def __init__(self, similarity_threshold: float = 0.6) -> None:
        self._threshold = similarity_threshold

    def evaluate(self, query: str, results: list[SearchResult]) -> EvalReport:
        query_tokens = _tokenize(query)
        evals: list[ResultEvaluation] = []

        for r in results:
            doc_tokens = _tokenize(f"{r.title or ''} {r.content or ''}")
            lexical_overlap = _jaccard(query_tokens, doc_tokens)
            term_coverage = _term_coverage(query_tokens, doc_tokens)
            relevant = r.similarity >= self._threshold and term_coverage > 0
            evals.append(
                ResultEvaluation(
                    product_id=r.product_id,
                    similarity=r.similarity,
                    lexical_overlap=lexical_overlap,
                    term_coverage=term_coverage,
                    relevant=relevant,
                )
            )

        k = len(results)
        mean_sim = sum(r.similarity for r in results) / k if k else 0.0
        relevant_count = sum(1 for e in evals if e.relevant)
        return EvalReport(
            query=query,
            k=k,
            threshold=self._threshold,
            results=evals,
            mean_similarity=mean_sim,
            relevant_count=relevant_count,
            relevance_ratio=relevant_count / k if k else 0.0,
        )


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _term_coverage(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)
