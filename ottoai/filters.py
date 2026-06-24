from domain.models.search import ProductFilters


def merge_filters(
    current: ProductFilters, delta: ProductFilters, reset: bool
) -> ProductFilters:
    """Accumulate filters across conversational turns.

    The conversational planner emits `delta` as only the filters this turn
    adds/changes (unchanged ones left null). We merge deterministically rather
    than trust the LLM to restate every active filter each turn:

    - `reset=True`  -> drop everything carried over; the new state is `delta`.
    - `reset=False` -> overlay `delta` onto `current`; non-null delta fields win,
      null delta fields keep the carried-over value.

    Known limitation: with `reset=False` a single null delta field cannot clear
    one carried filter (null means "unchanged"). Clearing all constraints goes
    through `reset=True`. This is an accepted v1 tradeoff in favour of a
    deterministic, predictable merge.
    """
    if reset:
        return delta.model_copy(deep=True)

    merged = current.model_dump()
    for field, value in delta.model_dump().items():
        if value is not None:
            merged[field] = value
    return ProductFilters(**merged)
