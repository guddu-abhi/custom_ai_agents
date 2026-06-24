import logging
import sys
from pathlib import Path

import click

from loader.config import settings
from loader.core.loader import LoadOrchestrator
from loader.db.embed_repo import EmbeddingRepository
from loader.db.engine import get_connection
from loader.db.product_repo import ProductRepository
from loader.utils.checkpoint import CheckpointManager
from otto_lib.embedding import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_LOADER_DIR = Path(__file__).parent
DATA_FILE = _LOADER_DIR / "meta_Electronics.jsonl"
CHECKPOINT_FILE = _LOADER_DIR / "loader_progress.json"


@click.group()
def cli() -> None:
    """Amazon product data loader CLI."""


@cli.command()
@click.option(
    "--env",
    default="local",
    show_default=True,
    type=click.Choice(["local", "dev", "qa", "prod"], case_sensitive=False),
    help="Target DB environment.",
)
@click.option(
    "--batch-size",
    default=settings.default_batch_size,
    show_default=True,
    type=int,
    help="Number of rows per DB commit.",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Ignore existing checkpoint and restart from line 0.",
)
@click.option(
    "--max-records",
    default=0,
    show_default=True,
    type=int,
    help="Stop after inserting this many records (0 = no limit).",
)
@click.option(
    "--no-embed",
    is_flag=True,
    default=False,
    help="Skip embedding generation (faster dry-run).",
)
def load(env: str, batch_size: int, reset: bool, max_records: int, no_embed: bool) -> None:
    """Stream the JSONL file and insert product rows into PostgreSQL."""
    if no_embed:
        embedder = None
        log.info("--no-embed: embedding generation disabled.")
    else:
        log.info("Using Ollama model %s at %s", settings.embed_model_name, settings.ollama_base_url)
        embedder = EmbeddingService(settings.embed_model_name, settings.ollama_base_url)

    checkpoint = CheckpointManager(CHECKPOINT_FILE)

    with get_connection(env) as conn:
        orchestrator = LoadOrchestrator(
            conn=conn,
            product_repo=ProductRepository(conn),
            embed_repo=EmbeddingRepository(conn),
            embedder=embedder,
            checkpoint=checkpoint,
        )
        orchestrator.run(
            data_file=DATA_FILE,
            batch_size=batch_size,
            reset=reset,
            max_records=max_records,
            min_year=settings.min_year,
        )
