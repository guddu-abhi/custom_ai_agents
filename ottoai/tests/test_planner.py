import os

import pytest

from agents import Runner
from ottoai.planner import planner_agent


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
async def test_planner_extracts_filters():
    result = await Runner.run(planner_agent, "Sony headphones under $50, 4 stars and up")
    plan = result.final_output
    assert plan.filters.brand is not None and "sony" in plan.filters.brand.lower()
    assert plan.filters.price_max == 50
    assert plan.filters.min_rating == 4
