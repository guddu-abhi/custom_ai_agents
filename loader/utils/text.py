import re

_SANITIZE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


def sanitize(value: str | None) -> str:
    """Remove ASCII control characters and collapse whitespace runs."""
    if not value:
        return ""
    value = _SANITIZE_RE.sub("", value)
    return _WS_RE.sub(" ", value).strip()
