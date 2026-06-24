import asyncio
import logging
import sys

import click
from dotenv import load_dotenv

from agents import Runner
from domain.models.search import OttoAnswer, SearchPlan
from ottoai.config import settings
from ottoai.otto_agent import OttoAgent
from ottoai.planner import planner_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Load root .env so the OpenAI SDK can read OPENAI_API_KEY itself.
load_dotenv()

_ENV_CHOICE = click.Choice(["local", "dev", "qa", "prod"], case_sensitive=False)


@click.command()
@click.argument("query", type=str)
@click.option("--env", default="local", show_default=True, type=_ENV_CHOICE)
def cli(query: str, env: str) -> None:
    """Plan -> retrieve -> answer over the product catalog (Otto AI)."""
    settings.env = env
    asyncio.run(_run(query))


async def _run(query: str) -> None:
    agent = OttoAgent()

    plan_result = await Runner.run(planner_agent, query)
    _print_plan(plan_result.final_output)

    answer = await agent.run(query)
    _print_answer(answer)
    _print_citations(answer)


def _print_plan(plan: SearchPlan) -> None:
    click.echo("")
    click.echo("=== plan ===")
    click.echo(f"query   : {plan.query}")
    f = plan.filters
    click.echo(
        f"filters : brand={f.brand} price_max={f.price_max} "
        f"min_rating={f.min_rating} min_reviews={f.min_reviews}"
    )
    click.echo("")


def _print_answer(answer: OttoAnswer) -> None:
    click.echo("=== answer ===")
    click.echo(answer.answer.strip() or "(empty answer)")
    click.echo("")


def _print_citations(answer: OttoAnswer) -> None:
    if not answer.citations:
        click.echo("(no citations parsed)")
        return
    header = f"{'#':>3}  {'pid':>8}  {'sim':>5}  title"
    click.echo(header)
    click.echo("-" * len(header))
    for i, c in enumerate(answer.citations, start=1):
        title = (c.title or "").replace("\n", " ").strip()
        if len(title) > 70:
            title = title[:69] + "…"
        click.echo(f"{i:>3}  {c.product_id:>8}  {c.similarity:>5.3f}  {title}")
