"""
data_loader.py — Production-ready Amazon product JSONL → PostgreSQL loader.

Usage:
    python -m loader.data_loader load [OPTIONS]

Options:
    --env          [local|dev|qa|prod]  Target DB environment (default: local)
    --batch-size   INTEGER              Rows per commit (default: 1000)
    --reset                             Ignore checkpoint; restart from line 0
"""

import json
import logging
import re
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Generator
import torch
import click
from sentence_transformers import SentenceTransformer
from sqlalchemy import String, bindparam, create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
LOADER_DIR = Path(__file__).parent
DATA_FILE = LOADER_DIR / "meta_Electronics.jsonl"
CHECKPOINT_FILE = LOADER_DIR / "loader_progress.json"
DEFAULT_BATCH_SIZE = 1000
MIN_YEAR = 2023

# Embedding model — Qwen3-Embedding-0.6B produces 1024-dim vectors, matching
# the catalog.product_embeddings.embedding vector(1024) column.
EMBED_MODEL_PATH = "./models/Qwen3-Embedding-0.6B"
EMBED_MODEL_NAME  = "Qwen3-Embedding-0.6B"

# Populated once at the start of the `load` command; None disables embedding.
_embedding_model: SentenceTransformer | None = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connection — sync psycopg3 via SQLAlchemy
# Mirrors the connection params in db_utils.WebAppDBFactory but uses the
# synchronous postgresql+psycopg dialect instead of asyncpg.
# ---------------------------------------------------------------------------
_DB_URLS: dict[str, str] = {
    "local": "postgresql+psycopg://postgres:example@localhost:5432/postgres",
    "dev":   "postgresql+psycopg://user:pass@dev-host:5432/dev_db",
    "qa":    "postgresql+psycopg://user:pass@qa-host:5432/qa_db",
    "prod":  "postgresql+psycopg://user:pass@prod-host:5432/prod_db",
}


def _get_engine(env: str):
    url = _DB_URLS.get(env)
    if not url:
        raise ValueError(f"Unknown environment {env!r}. Choices: {list(_DB_URLS)}")
    return create_engine(url, echo=False, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers (file-based)
# ---------------------------------------------------------------------------
def _load_checkpoint() -> int:
    """Return the last successfully committed line number, or 0 if no checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            return int(json.loads(CHECKPOINT_FILE.read_text()).get("last_committed_line", 0))
        except Exception:
            pass
    return 0


def _save_checkpoint(last_line: int) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"last_committed_line": last_line}, indent=2)
    )


# ---------------------------------------------------------------------------
# JSONL reader — memory-efficient, line-by-line
# ---------------------------------------------------------------------------
def read_jsonl(
    file_path: Path, start_line: int = 0
) -> Generator[tuple[int, dict], None, None]:
    """
    Yield ``(line_number, parsed_record)`` for every valid JSON line in *file_path*.

    Lines numbered below *start_line* are skipped (supports resumability).
    Malformed JSON lines are logged and skipped without aborting the run.
    """
    with file_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh):
            if line_no < start_line:
                continue
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSON at line %d: %s", line_no, exc)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

# Strip ASCII control characters (keep printable Unicode + whitespace).
_SANITIZE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE        = re.compile(r"\s+")


def _sanitize(text: str | None) -> str:
    """Remove control characters and collapse runs of whitespace."""
    if not text:
        return ""
    text = _SANITIZE_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DATE_FMTS = ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y")


def _parse_year(details: dict) -> int | None:
    """Extract a 4-digit year from the 'Date First Available' field in *details*."""
    raw = (details.get("Date First Available") or "").strip()
    if not raw:
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(raw, fmt).year
        except ValueError:
            continue
    # Fallback: grab the last plausible year token in the string
    matches = _YEAR_RE.findall(raw)
    return int(matches[-1]) if matches else None


def _parse_price(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None


def transform(record: dict) -> dict | None:
    """
    Map a raw JSONL record to a ``catalog.products`` row dict.

    Returns ``None`` when the record should be filtered out (pre-2020 or missing
    required fields).
    """
    details = record.get("details") or {}
    if not isinstance(details, dict):
        details = {}

    # ---- Date filter: keep only records from MIN_YEAR onwards ----
    year = _parse_year(details)
    if year is not None and year < MIN_YEAR:
        return None

    # ---- description: join list elements with newline ----
    desc = record.get("description")
    if isinstance(desc, list):
        description: str | None = "\n".join(str(d) for d in desc if d) or None
    else:
        description = desc or None

    # ---- TEXT[] columns: ensure list ----
    features = record.get("features")
    if not isinstance(features, list):
        features = []

    categories = record.get("categories")
    if not isinstance(categories, list):
        categories = []

    # ---- embedding content fields (sanitized plain-text) ----
    # tier_1 ("discovery"): broad signal — category + title + store
    # tier_2 ("detail"):    rich signal  — title + description + features
    tier_1_content = _sanitize(" ".join(filter(None, [
        record.get("main_category"),
        record.get("title"),
        record.get("store"),
    ])))
    tier_2_content = _sanitize(" ".join(filter(None, [
        record.get("title"),
        description,
        " ".join(features) if features else None,
    ])))

    return {
        "main_category":           record.get("main_category"),
        "title":                   record.get("title"),
        "average_rating":          record.get("average_rating") or 0,
        "rating_number":           record.get("rating_number") or 0,
        "features":                features,
        "description":             description,
        "price":                   _parse_price(record.get("price")),
        "store":                   record.get("store"),
        "categories":              categories,
        "details":                 details,
        "parent_asin":             record.get("parent_asin") or "",
        "tier_1_embedding_content": tier_1_content,
        "tier_2_embedding_content": tier_2_content,
    }


# ---------------------------------------------------------------------------
# Prepared INSERT statement
# bindparam type annotations let SQLAlchemy/psycopg3 serialise Python
# lists → TEXT[]  and  Python dicts → JSONB correctly.
# ---------------------------------------------------------------------------
_INSERT_SQL = text(
    """
    INSERT INTO catalog.products (
        main_category, title, average_rating, rating_number,
        features, description, price, store,
        categories, details, parent_asin
    ) VALUES (
        :main_category, :title, :average_rating, :rating_number,
        :features, :description, :price, :store,
        :categories, :details, :parent_asin
    )
    RETURNING id
    """
).bindparams(
    bindparam("features",   type_=ARRAY(String)),
    bindparam("categories", type_=ARRAY(String)),
    bindparam("details",    type_=JSONB()),
)

# ---------------------------------------------------------------------------
# Embedding INSERT — upserts on (product_id, tier, model_name) so re-runs are
# idempotent.  The ::catalog.embed_tier cast satisfies psycopg3's ENUM check.
# ---------------------------------------------------------------------------
_EMBED_INSERT_SQL = text(
    """
    INSERT INTO catalog.product_embeddings
        (product_id, tier, model_name, content, embedding)
    VALUES
        (:product_id, CAST(:tier AS catalog.embed_tier), :model_name, :content, CAST(:embedding AS vector))
    ON CONFLICT (product_id, tier, model_name)
    DO UPDATE SET
        content   = EXCLUDED.content,
        embedding = EXCLUDED.embedding
    """
)


# ---------------------------------------------------------------------------
# Batch commit (extracted to keep the main loop readable)
# ---------------------------------------------------------------------------
def _commit_batch(conn, batch, line_no, stats):
    tier1_texts = [row.pop("tier_1_embedding_content", "") for row in batch]
    tier2_texts = [row.pop("tier_2_embedding_content", "") for row in batch]

    product_ids: list[int] = []
    try:
        for row in batch:
            result = conn.execute(_INSERT_SQL, row)
            product_ids.append(result.scalar_one())
        conn.commit()
        stats["inserted"] += len(batch)
    except Exception as exc:
        conn.rollback()
        log.error("Batch insert failed near line %d ...", line_no, len(batch), exc)
        stats["errors"] += len(batch)
        batch.clear()   # ← clear and return early on failure
        return          # ← don't attempt embeddings with no product_ids
    
    batch.clear()       # ← clear after confirmed success

    if not product_ids or _embedding_model is None:
        return
    # ---- Generate + store embeddings ----
    try:
        tier1_embs = _embedding_model.encode(
            tier1_texts, convert_to_numpy=True, show_progress_bar=False
        )
        tier2_embs = _embedding_model.encode(
            tier2_texts, convert_to_numpy=True, show_progress_bar=False
        )

        emb_rows = []
        for pid, t1_text, t1_emb, t2_text, t2_emb in zip(
            product_ids, tier1_texts, tier1_embs, tier2_texts, tier2_embs
        ):
            emb_rows.append({
                "product_id": pid,
                "tier":       "discovery",
                "model_name": EMBED_MODEL_NAME,
                "content":    t1_text,
                "embedding":  str(t1_emb.tolist()),
            })
            emb_rows.append({
                "product_id": pid,
                "tier":       "detail",
                "model_name": EMBED_MODEL_NAME,
                "content":    t2_text,
                "embedding":  str(t2_emb.tolist()),
            })

        conn.execute(_EMBED_INSERT_SQL, emb_rows)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        log.error("Embedding insert failed near line %d: %s", line_no, exc)
        stats["errors"] += len(product_ids) * 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
    default=DEFAULT_BATCH_SIZE,
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
def load(env: str, batch_size: int, reset: bool, max_records: int) -> None:
    """Stream the JSONL file and insert product rows into PostgreSQL."""
    global _embedding_model

    # ---- Load embedding model (once) ----
    log.info("Loading embedding model from %s …", EMBED_MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _embedding_model = SentenceTransformer(EMBED_MODEL_PATH, device=device)    
    log.info("Embedding model ready.")

    # ---- Checkpoint ----
    if reset:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        log.info("--reset: ignoring checkpoint, starting from line 0.")
        start_line = 0
    else:
        start_line = _load_checkpoint()
        if start_line:
            log.info("Resuming from line %d (checkpoint found).", start_line)

    # ---- Connect ----
    engine = _get_engine(env)
    log.info("Target env=%s | data file=%s", env, DATA_FILE)

    stats = {"processed": 0, "inserted": 0, "filtered": 0, "errors": 0}
    batch: list[dict] = []
    last_line = start_line
    done = False

    if max_records > 0:
        log.info("--max-records=%d: will stop after inserting %d records.", max_records, max_records)

    with engine.connect() as conn:
        for line_no, record in read_jsonl(DATA_FILE, start_line=start_line):
            last_line = line_no
            stats["processed"] += 1

            try:
                row = transform(record)
            except Exception as exc:
                log.warning("Transform error at line %d: %s", line_no, exc)
                stats["errors"] += 1
                continue

            if row is None:
                stats["filtered"] += 1
                continue

            # Trim the row to stay within the max-records cap before batching.
            if max_records > 0:
                remaining = max_records - stats["inserted"] - len(batch)
                if remaining <= 0:
                    done = True
                    break

            batch.append(row)

            if len(batch) >= batch_size:
                _commit_batch(conn, batch, line_no, stats)
                _save_checkpoint(last_line)
                log.info(
                    "line=%-8d  inserted=%-8d  filtered=%-6d  errors=%d",
                    last_line, stats["inserted"], stats["filtered"], stats["errors"],
                )
                if max_records > 0 and stats["inserted"] >= max_records:
                    done = True
                    break

        # ---- Flush the final partial batch ----
        if batch:
            _commit_batch(conn, batch, last_line, stats)
            _save_checkpoint(last_line)

    if done:
        log.info("--max-records limit (%d) reached — stopping early.", max_records)

    log.info(
        "Load complete. processed=%d  inserted=%d  filtered=%d  errors=%d",
        stats["processed"], stats["inserted"], stats["filtered"], stats["errors"],
    )


if __name__ == "__main__":
    cli()
