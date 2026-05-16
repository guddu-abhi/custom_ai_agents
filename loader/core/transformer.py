import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from loader.utils.text import sanitize

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DATE_FMTS = ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y")


@dataclass
class PipelineRow:
    db_row: dict
    embedding_content: str


def _parse_year(details: dict) -> int | None:
    raw = (details.get("Date First Available") or "").strip()
    if not raw:
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(raw, fmt).year
        except ValueError:
            continue
    matches = _YEAR_RE.findall(raw)
    return int(matches[-1]) if matches else None


def _parse_price(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None


def transform(record: dict, min_year: int = 2023) -> PipelineRow | None:
    """Map a raw JSONL record to a PipelineRow. Returns None to filter the record."""
    details = record.get("details") or {}
    if not isinstance(details, dict):
        details = {}

    year = _parse_year(details)
    if year is not None and year < min_year:
        return None

    desc = record.get("description")
    if isinstance(desc, list):
        description: str | None = "\n".join(str(d) for d in desc if d) or None
    else:
        description = desc or None

    features = record.get("features")
    if not isinstance(features, list):
        features = []

    categories = record.get("categories")
    if not isinstance(categories, list):
        categories = []

    # First 4 features as clean text (skip items that look like raw JSON blobs)
    feature_snippets = [
        f[:200]
        for f in (features or [])
        if isinstance(f, str) and not f.lstrip().startswith("{")
    ][:4]
    feature_text = " ".join(feature_snippets) or None

    # Truncate description — the full text often contains noisy JSON blobs
    desc_snippet = (description or "")[:600] or None

    embedding_content = sanitize(" ".join(filter(None, [
        record.get("main_category"),
        record.get("title"),
        record.get("store"),
        feature_text,
        desc_snippet,
    ])))[:3500]  # hard cap: keeps well within nomic-embed-text's 2048-token default

    return PipelineRow(
        db_row={
            "main_category":  record.get("main_category"),
            "title":          record.get("title"),
            "average_rating": record.get("average_rating") or 0,
            "rating_number":  record.get("rating_number") or 0,
            "features":       features,
            "description":    description,
            "price":          _parse_price(record.get("price")),
            "store":          record.get("store"),
            "categories":     categories,
            "details":        details,
            "parent_asin":    record.get("parent_asin") or "",
        },
        embedding_content=embedding_content,
    )
