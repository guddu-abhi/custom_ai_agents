import asyncio
import logging
import sys

import click

from domain.models.search import SearchResult
from loader.config import settings as loader_settings
from otto_lib.config import Env
from otto_lib.db.engine import WebAppDBFactory
from otto_lib.embedding import EmbeddingService
from retrieval.config import settings as retrieval_settings
from retrieval.core.evaluator import EvalReport, EvaluationService
from retrieval.core.searcher import SearchService
from retrieval.db.search_repo import SearchRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_ENV_CHOICE = click.Choice(["local", "dev", "qa", "prod"], case_sensitive=False)


@click.group()
def cli() -> None:
    """Retrieval CLI: vector search + lightweight relevance eval."""


@cli.command()
@click.argument("query", type=str)
@click.option("--env", default="local", show_default=True, type=_ENV_CHOICE)
@click.option(
    "--k",
    default=retrieval_settings.default_k,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of results to return.",
)
def search(query: str, env: str, k: int) -> None:
    """Embed QUERY and return top-k matching products by cosine similarity."""
    log.info("Embedding query via %s @ %s", loader_settings.embed_model_name, loader_settings.ollama_base_url)
    embedder = EmbeddingService(loader_settings.embed_model_name, loader_settings.ollama_base_url)

    async def _retrieve() -> list[SearchResult]:
        async with WebAppDBFactory.get_db_engine(Env(env)).connect() as conn:
            return await SearchService(embedder, SearchRepository(conn)).search(query, k=k)

    results = asyncio.run(_retrieve())

    _print_results_table(results)
    _print_ids_tuple(results)


@cli.command(name="eval")
@click.argument("query", type=str)
@click.option("--env", default="local", show_default=True, type=_ENV_CHOICE)
@click.option(
    "--k",
    default=retrieval_settings.default_k,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of results to return.",
)
@click.option(
    "--threshold",
    default=retrieval_settings.similarity_threshold,
    show_default=True,
    type=float,
    help="Cosine similarity threshold for the relevance gate.",
)
def eval_cmd(query: str, env: str, k: int, threshold: float) -> None:
    """Run search + heuristic relevance evaluation on QUERY."""
    log.info("Embedding query via %s @ %s", loader_settings.embed_model_name, loader_settings.ollama_base_url)
    embedder = EmbeddingService(loader_settings.embed_model_name, loader_settings.ollama_base_url)
    evaluator = EvaluationService(similarity_threshold=threshold)

    async def _retrieve() -> list[SearchResult]:
        async with WebAppDBFactory.get_db_engine(Env(env)).connect() as conn:
            return await SearchService(embedder, SearchRepository(conn)).search(query, k=k)

    results = asyncio.run(_retrieve())

    report = evaluator.evaluate(query, results)
    _print_eval_table(results, report)
    _print_eval_summary(report)
    _print_ids_tuple(results)


def _truncate(s: str | None, n: int) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_results_table(results: list[SearchResult]) -> None:
    if not results:
        click.echo("(no results)")
        return
    header = f"{'#':>3}  {'sim':>5}  {'id':>8}  {'category':<22}  {'price':>8}  {'rating':>6}  title"
    click.echo(header)
    click.echo("-" * len(header))
    for i, r in enumerate(results, start=1):
        price = f"{r.price:.2f}" if r.price is not None else "-"
        click.echo(
            f"{i:>3}  {r.similarity:>5.3f}  {r.product_id:>8}  "
            f"{_truncate(r.main_category, 22):<22}  {price:>8}  "
            f"{float(r.average_rating):>6.2f}  {_truncate(r.title, 70)}"
        )


def _print_eval_table(results: list[SearchResult], report: EvalReport) -> None:
    if not results:
        click.echo("(no results)")
        return
    header = (
        f"{'#':>3}  {'sim':>5}  {'lex':>5}  {'cov':>5}  {'rel':>3}  "
        f"{'id':>8}  {'category':<22}  title"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for i, (r, e) in enumerate(zip(results, report.results), start=1):
        click.echo(
            f"{i:>3}  {e.similarity:>5.3f}  {e.lexical_overlap:>5.3f}  "
            f"{e.term_coverage:>5.3f}  {'Y' if e.relevant else 'N':>3}  "
            f"{r.product_id:>8}  {_truncate(r.main_category, 22):<22}  "
            f"{_truncate(r.title, 60)}"
        )


def _print_eval_summary(report: EvalReport) -> None:
    click.echo("")
    click.echo(
        f"query={report.query!r}  k={report.k}  threshold={report.threshold:.2f}  "
        f"mean_sim={report.mean_similarity:.3f}  "
        f"relevant={report.relevant_count}/{report.k}  "
        f"ratio={report.relevance_ratio:.2f}"
    )


def _print_ids_tuple(results: list[SearchResult]) -> None:
    if not results:
        return
    ids = tuple(r.product_id for r in results)
    click.echo("")
    click.echo(f"product_ids: {ids}")
    click.echo(f"-- SQL: SELECT * FROM catalog.products WHERE id IN {ids};")
