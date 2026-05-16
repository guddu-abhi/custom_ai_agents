import hashlib
import json
import logging
import re
from pathlib import Path

import nltk

from domain.models.generation import GenerationMetrics, GenerationResult
from generation.core.prompt import CITATION_RE
from generation.core.providers.base import GenerationParams
from generation.core.providers.openai import OpenAIProvider
from retrieval.db.search_repo import SearchResult

log = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are a strict evaluator of product recommendation answers. "
    "Given a user QUESTION, an ANSWER, and the TITLES of products that were "
    "retrieved as candidates, rate how well the answer addresses the question "
    "given those candidates. Score on a 0–2 integer rubric:\n"
    "  0 = unhelpful, off-topic, or fabricated\n"
    "  1 = partially helpful but with gaps or weak grounding\n"
    "  2 = clearly helpful and grounded in the candidates\n"
    "Respond with ONLY a single digit: 0, 1, or 2."
)

_CACHE_DIR = Path.home() / ".cache" / "custom_ai_agents" / "judge"


class EvaluationService:
    def __init__(self, judge_enabled: bool = False, judge_model: str = "gpt-5-nano") -> None:
        self._judge_enabled = judge_enabled
        self._judge_model = judge_model
        self._tokenizer = _load_punkt()

    def evaluate(
        self,
        query: str,
        results: list[SearchResult],
        generation: GenerationResult,
    ) -> GenerationMetrics:
        cited_pids = _cited_pids(generation.answer)
        retrieved_pids = {r.product_id for r in results}

        if cited_pids:
            faithfulness = len(cited_pids & retrieved_pids) / len(cited_pids)
        else:
            faithfulness = 0.0

        cited_unknown = tuple(sorted(cited_pids - retrieved_pids))
        citation_coverage = _citation_coverage(generation.answer, self._tokenizer)

        judge_score: float | None = None
        if self._judge_enabled:
            judge_score = self._run_judge(query, generation, results)

        return GenerationMetrics(
            faithfulness=faithfulness,
            citation_coverage=citation_coverage,
            cited_unknown=cited_unknown,
            latency_ms=generation.latency_ms,
            prompt_tokens=generation.usage.prompt_tokens,
            completion_tokens=generation.usage.completion_tokens,
            judge_score=judge_score,
        )

    def _run_judge(
        self,
        query: str,
        generation: GenerationResult,
        results: list[SearchResult],
    ) -> float | None:
        titles = [r.title or "" for r in results]
        prompt_hash = _hash_prompt(query, generation.answer, titles)
        cache_key = f"{query}::{self._judge_model}::{prompt_hash}"
        cached = _judge_cache_get(cache_key)
        if cached is not None:
            return cached

        user = (
            f"QUESTION:\n{query}\n\n"
            f"ANSWER:\n{generation.answer}\n\n"
            f"RETRIEVED PRODUCT TITLES:\n" + "\n".join(f"- {t}" for t in titles)
        )
        provider = OpenAIProvider()
        try:
            resp = provider.complete(
                _JUDGE_SYSTEM,
                user,
                GenerationParams(model=self._judge_model, temperature=0.0, max_tokens=4),
            )
        except Exception as exc:  # pragma: no cover - network errors
            log.warning("Judge call failed: %s", exc)
            return None

        score = _parse_judge_score(resp.text)
        if score is not None:
            _judge_cache_put(cache_key, score)
        return score


def _cited_pids(answer: str) -> set[int]:
    return {int(m.group(1)) for m in CITATION_RE.finditer(answer)}


def _citation_coverage(answer: str, tokenizer) -> float:  # type: ignore[no-untyped-def]
    if not answer.strip():
        return 0.0
    sentences = tokenizer.tokenize(answer)
    if not sentences:
        return 0.0
    with_cite = sum(1 for s in sentences if CITATION_RE.search(s))
    return with_cite / len(sentences)


def _hash_prompt(query: str, answer: str, titles: list[str]) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\x00")
    h.update(answer.encode("utf-8"))
    h.update(b"\x00")
    h.update("\n".join(titles).encode("utf-8"))
    return h.hexdigest()[:16]


def _judge_cache_path(key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{h}.json"


def _judge_cache_get(key: str) -> float | None:
    path = _judge_cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        v = data.get("score")
        return float(v) if v is not None else None
    except Exception:
        return None


def _judge_cache_put(key: str, score: float) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _judge_cache_path(key).write_text(json.dumps({"score": score}))


_SCORE_RE = re.compile(r"[0-2]")


def _parse_judge_score(text: str) -> float | None:
    m = _SCORE_RE.search(text.strip())
    return float(m.group(0)) if m else None


class _PunktTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return nltk.sent_tokenize(text)


def _load_punkt() -> _PunktTokenizer:
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return _PunktTokenizer()
        except LookupError:
            continue
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.download(pkg, quiet=True)
            nltk.data.find(f"tokenizers/{pkg}")
            return _PunktTokenizer()
        except LookupError:
            continue
    raise LookupError("Could not load NLTK punkt tokenizer")
