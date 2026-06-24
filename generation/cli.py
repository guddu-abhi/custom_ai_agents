import asyncio
import logging
import sys

import click
from dotenv import load_dotenv

from domain.models.generation import GenerationMetrics, GenerationResult, ProviderName
from domain.models.search import SearchResult
from generation.config import settings as gen_settings
from generation.core.evaluator import EvaluationService
from generation.core.generator import GenerationService
from generation.core.prompt import PromptBuilder
from loader.config import settings as loader_settings
from otto_lib.config import Env
from otto_lib.db.engine import WebAppDBFactory
from otto_lib.embedding import EmbeddingService
from otto_lib.llm import get_provider
from otto_lib.llm.base import GenerationParams
from retrieval.core.searcher import SearchService
from retrieval.db.search_repo import SearchRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Load root .env into the process environment so the OpenAI SDK can read
# OPENAI_API_KEY itself (per generation/CLAUDE.md, the provider must not read it).
load_dotenv()

_ENV_CHOICE = click.Choice(["local", "dev", "qa", "prod"], case_sensitive=False)
_PROVIDER_CHOICE = click.Choice(["openai", "ollama"], case_sensitive=False)


@click.group()
def cli() -> None:
    """Generation CLI: grounded RAG answers + LLM grounding eval."""


@cli.command()
@click.argument("query", type=str)
@click.option("--env", default="local", show_default=True, type=_ENV_CHOICE)
@click.option(
    "--provider",
    default=gen_settings.provider,
    show_default=True,
    type=_PROVIDER_CHOICE,
)
@click.option("--model", default="", help="Override model name; otherwise provider default is used.")
@click.option(
    "--k",
    default=gen_settings.default_k,
    show_default=True,
    type=click.IntRange(min=1),
    help="Docs retrieved & fed to the LLM.",
)
@click.option(
    "--temperature",
    default=gen_settings.temperature,
    show_default=True,
    type=float,
)
@click.option(
    "--max-tokens",
    default=gen_settings.max_tokens,
    show_default=True,
    type=click.IntRange(min=1),
)
def generate(
    query: str,
    env: str,
    provider: str,
    model: str,
    k: int,
    temperature: float,
    max_tokens: int,
) -> None:
    """Retrieve, prompt an LLM, print a grounded answer + citations."""
    results, gen = _run_pipeline(
        query=query,
        env=env,
        provider=provider,
        model=model,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _print_answer(gen)
    _print_citation_table(gen, results)
    _print_usage(gen)


@cli.command(name="rag-eval")
@click.argument("query", type=str)
@click.option("--env", default="local", show_default=True, type=_ENV_CHOICE)
@click.option(
    "--provider",
    default=gen_settings.provider,
    show_default=True,
    type=_PROVIDER_CHOICE,
)
@click.option("--model", default="", help="Override model name; otherwise provider default is used.")
@click.option(
    "--k",
    default=gen_settings.default_k,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option(
    "--threshold",
    default=0.6,
    show_default=True,
    type=float,
    help="(Informational) similarity threshold reported alongside metrics.",
)
@click.option("--judge", is_flag=True, default=False, help="Also run LLM-as-judge 0–2 scoring.")
@click.option(
    "--judge-model",
    default="gpt-5-nano",
    show_default=True,
    help="Small OpenAI model used for --judge.",
)
@click.option(
    "--temperature",
    default=gen_settings.temperature,
    show_default=True,
    type=float,
)
@click.option(
    "--max-tokens",
    default=gen_settings.max_tokens,
    show_default=True,
    type=click.IntRange(min=1),
)
def rag_eval(
    query: str,
    env: str,
    provider: str,
    model: str,
    k: int,
    threshold: float,
    judge: bool,
    judge_model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    """Run full RAG pipeline and compute grounding metrics."""
    results, gen = _run_pipeline(
        query=query,
        env=env,
        provider=provider,
        model=model,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    evaluator = EvaluationService(judge_enabled=judge, judge_model=judge_model)
    metrics = evaluator.evaluate(query, results, gen)

    _print_answer(gen)
    _print_citation_table(gen, results)
    _print_usage(gen)
    _print_metrics(metrics, threshold)


def _run_pipeline(
    query: str,
    env: str,
    provider: str,
    model: str,
    k: int,
    temperature: float,
    max_tokens: int,
) -> tuple[list[SearchResult], GenerationResult]:
    provider_name: ProviderName = provider  # type: ignore[assignment]
    chosen_model = model or _default_model(provider_name)

    log.info(
        "Embedding query via %s @ %s",
        loader_settings.embed_model_name,
        loader_settings.ollama_base_url,
    )
    embedder = EmbeddingService(
        loader_settings.embed_model_name, loader_settings.ollama_base_url
    )

    async def _retrieve() -> list[SearchResult]:
        async with WebAppDBFactory.get_db_engine(Env(env)).connect() as conn:
            return await SearchService(embedder, SearchRepository(conn)).search(query, k=k)

    results = asyncio.run(_retrieve())

    log.info("Generating answer via provider=%s model=%s", provider_name, chosen_model)
    llm = get_provider(provider_name)
    prompt_builder = PromptBuilder(max_context_chars=gen_settings.max_context_chars)
    params = GenerationParams(
        model=chosen_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    service = GenerationService(llm, prompt_builder, params)
    gen = service.answer(query, results)
    return results, gen


def _default_model(provider: ProviderName) -> str:
    return gen_settings.openai_model if provider == "openai" else gen_settings.ollama_model


def _truncate(s: str | None, n: int) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_answer(gen: GenerationResult) -> None:
    click.echo("")
    click.echo(f"=== answer (provider={gen.provider} model={gen.model}) ===")
    click.echo(gen.answer.strip() or "(empty answer)")
    click.echo("")


def _print_citation_table(gen: GenerationResult, results: list[SearchResult]) -> None:
    if not gen.citations:
        click.echo("(no citations parsed)")
        return
    header = f"{'#':>3}  {'pid':>8}  {'sim':>5}  title"
    click.echo(header)
    click.echo("-" * len(header))
    for i, c in enumerate(gen.citations, start=1):
        click.echo(f"{i:>3}  {c.product_id:>8}  {c.similarity:>5.3f}  {_truncate(c.title, 70)}")
    click.echo("")


def _print_usage(gen: GenerationResult) -> None:
    click.echo(
        f"latency_ms={gen.latency_ms}  "
        f"prompt_tokens={gen.usage.prompt_tokens}  "
        f"completion_tokens={gen.usage.completion_tokens}  "
        f"total_tokens={gen.usage.total_tokens}"
    )


def _print_metrics(m: GenerationMetrics, threshold: float) -> None:
    click.echo("")
    click.echo("=== metrics ===")
    click.echo(f"faithfulness        : {m.faithfulness:.3f}")
    click.echo(f"citation_coverage   : {m.citation_coverage:.3f}")
    click.echo(f"cited_unknown       : {m.cited_unknown}")
    click.echo(f"latency_ms          : {m.latency_ms}")
    click.echo(f"prompt_tokens       : {m.prompt_tokens}")
    click.echo(f"completion_tokens   : {m.completion_tokens}")
    if m.judge_score is not None:
        click.echo(f"judge_score (0-2)   : {m.judge_score:.1f}")
    click.echo(f"(reported similarity threshold: {threshold:.2f})")
