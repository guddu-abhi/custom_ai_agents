"""Read/write conversational state on top of the Agents SDK `Session`.

We deliberately use the SDK `Session` (a `SQLAlchemySession` in the webapp) as a
*manual* transcript + state store via `get_items` / `add_items` — NOT via the
`Runner(session=...)` auto-memory. Otto's answerer input is a big RAG-context
blob (`PromptBuilder` output); letting the Runner persist that as the "user
message" would pollute history and explode tokens on the next turn. Here we
control exactly what is written: clean user/assistant text turns plus one JSON
state item.
"""

from agents import TResponseInputItem
from agents.memory import Session
from domain.models.search_plan import OttoTurnState

# Marker prefix that tags the single structured-state item among the message
# items, so we can pick it out (and filter it from the human transcript).
_STATE_PREFIX = "\x00otto_state:"


async def load_state(session: Session) -> OttoTurnState:
    """Return the most recently persisted turn state, or a fresh empty one."""
    items = await session.get_items()
    for item in reversed(items):
        content = item.get("content")
        if isinstance(content, str) and content.startswith(_STATE_PREFIX):
            return OttoTurnState.model_validate_json(content[len(_STATE_PREFIX):])
    return OttoTurnState()


async def load_transcript(session: Session, max_turns: int) -> list[tuple[str, str]]:
    """Return the recent human-readable transcript as (role, content) pairs,
    excluding the internal state items. Trimmed to the last `max_turns` exchanges
    (a user+assistant pair counts as one turn)."""
    items = await session.get_items()
    pairs: list[tuple[str, str]] = []
    for item in items:
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if content.startswith(_STATE_PREFIX):
            continue
        pairs.append((role, content))
    return pairs[-(2 * max_turns):]


def render_transcript(pairs: list[tuple[str, str]]) -> str:
    label = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{label.get(r, r)}: {c}" for r, c in pairs)


async def append_turn(
    session: Session,
    user_message: str,
    assistant_message: str,
    state: OttoTurnState,
) -> None:
    """Persist one completed turn: the clean user + assistant messages, then the
    updated structured state as a tagged item (latest wins on read)."""
    items: list[TResponseInputItem] = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "system", "content": _STATE_PREFIX + state.model_dump_json()},
    ]
    await session.add_items(items)
