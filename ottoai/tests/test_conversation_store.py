"""Unit tests for the transcript/state store logic using an in-memory fake
Session (no DB). Exercises state-marker filtering and latest-wins reads."""

from domain.models.search import OttoTurnState, ProductFilters, ShownProduct
from ottoai.conversation_store import (
    append_turn,
    load_state,
    load_transcript,
    render_transcript,
)


class FakeSession:
    """Minimal in-memory stand-in for the SDK Session protocol."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    async def get_items(self, limit=None):
        return list(self.items) if limit is None else list(self.items[-limit:])

    async def add_items(self, items):
        self.items.extend(items)


async def test_empty_session_returns_default_state():
    s = FakeSession()
    state = await load_state(s)
    assert state.filters.model_dump(exclude_none=True) == {}
    assert state.shown_products == []
    assert await load_transcript(s, max_turns=6) == []


async def test_append_then_load_roundtrip():
    s = FakeSession()
    state = OttoTurnState(
        filters=ProductFilters(price_max=40),
        shown_products=[ShownProduct(product_id=7, title="Thing")],
    )
    await append_turn(s, "find headphones", "here you go [pid:7]", state)

    loaded = await load_state(s)
    assert loaded.filters.price_max == 40
    assert loaded.shown_products[0].product_id == 7

    pairs = await load_transcript(s, max_turns=6)
    assert pairs == [("user", "find headphones"), ("assistant", "here you go [pid:7]")]


async def test_latest_state_wins_and_transcript_excludes_state_items():
    s = FakeSession()
    await append_turn(s, "u1", "a1", OttoTurnState(filters=ProductFilters(price_max=100)))
    await append_turn(s, "u2", "a2", OttoTurnState(filters=ProductFilters(price_max=40)))

    # latest state wins
    assert (await load_state(s)).filters.price_max == 40
    # transcript holds only human turns, no state items
    pairs = await load_transcript(s, max_turns=6)
    assert pairs == [("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")]


async def test_transcript_trims_to_max_turns():
    s = FakeSession()
    for i in range(5):
        await append_turn(s, f"u{i}", f"a{i}", OttoTurnState())
    pairs = await load_transcript(s, max_turns=2)
    # 2 turns => last 4 messages
    assert pairs == [("user", "u3"), ("assistant", "a3"), ("user", "u4"), ("assistant", "a4")]


def test_render_transcript_format():
    rendered = render_transcript([("user", "hi"), ("assistant", "hello")])
    assert rendered == "User: hi\nAssistant: hello"
