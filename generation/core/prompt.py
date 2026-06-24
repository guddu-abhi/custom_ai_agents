import re
from dataclasses import dataclass

from domain.models.search import SearchResult

CITATION_RE = re.compile(r"\[pid:(\d+)\]")

SYSTEM_PROMPT = (
    "You are a product recommendation assistant. Answer the user's question using "
    "ONLY the products listed below. Cite every product you mention using the "
    "exact form `[pid:<id>]` (e.g. `[pid:42]`). If the listed products do not "
    "answer the question, say so explicitly — do not invent products, SKUs, or "
    "features. Keep answers under 100 sentences unless the user asks for detail."
)

_TRUNCATED_NOTICE = (
    "\n\nNOTE: Product context was truncated to fit the prompt budget; "
    "some product details are partial (indicated with '…')."
)


@dataclass(frozen=True)
class PromptPayload:
    system: str
    user: str


class PromptBuilder:
    def __init__(self, max_context_chars: int = 6000) -> None:
        self._max_context_chars = max_context_chars

    def build(self, query: str, results: list[SearchResult]) -> PromptPayload:
        header = f"QUESTION: {query}\n\nCANDIDATE PRODUCTS:\n"
        budget = self._max_context_chars - len(header)
        block, truncated = _render_products(results, budget)

        system = SYSTEM_PROMPT
        user = header + block
        if truncated:
            user += _TRUNCATED_NOTICE
        return PromptPayload(system=system, user=user)


def _render_products(results: list[SearchResult], budget: int) -> tuple[str, bool]:
    if budget <= 0:
        return "", True

    chunks: list[str] = []
    used = 0
    truncated = False

    for r in results:
        title = (r.title or "").strip().replace("\n", " ")
        category = (r.main_category or "").strip()
        price = f"{r.price:.2f}" if r.price is not None else "-"
        rating = f"{float(r.average_rating):.2f}"
        features = (r.description or r.content or "").strip().replace("\n", " ")

        head = f"[pid:{r.product_id}] {title} — {category}\n"
        meta = f"  price: {price}  rating: {rating}\n"
        per_product_overhead = len(head) + len(meta) + len("  features: \n")
        remaining = budget - used - per_product_overhead

        if remaining <= 0:
            truncated = True
            break

        if len(features) > remaining:
            features = features[: max(0, remaining - 1)] + "…"
            truncated = True

        line = f"{head}{meta}  features: {features}\n"
        used += len(line)
        chunks.append(line)

        if used >= budget:
            break

    return "".join(chunks), truncated
