import json
import logging
import sys
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Connection

from loader.core.transformer import PipelineRow, transform
from loader.db.embed_repo import EmbeddingRepository
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


@dataclass
class LoadStats:
    processed: int = 0
    inserted: int = 0
    filtered: int = 0
    errors: int = 0

    def __str__(self) -> str:
        return (
            f"processed={self.processed}  inserted={self.inserted}"
            f"  filtered={self.filtered}  errors={self.errors}"
        )


def _read_jsonl(
    file_path: Path, start_line: int = 0
) -> Generator[tuple[int, dict], None, None]:
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


class LoadOrchestrator:
    def __init__(
        self,
        conn: Connection,
        product_repo: ProductRepository,
        embed_repo: EmbeddingRepository,
        embedder: EmbeddingService | None,
        checkpoint: CheckpointManager,
    ) -> None:
        self._conn = conn
        self._product_repo = product_repo
        self._embed_repo = embed_repo
        self._embedder = embedder
        self._checkpoint = checkpoint

    def run(
        self,
        data_file: Path,
        batch_size: int = 1000,
        reset: bool = False,
        max_records: int = 0,
        min_year: int = 2023,
    ) -> LoadStats:
        stats = LoadStats()

        if reset:
            self._checkpoint.clear()
            log.info("--reset: ignoring checkpoint, starting from line 0.")
            start_line = 0
        else:
            start_line = self._checkpoint.load()
            if start_line:
                log.info("Resuming from line %d (checkpoint found).", start_line)

        if max_records > 0:
            log.info("--max-records=%d: will stop after inserting %d records.", max_records, max_records)

        batch: list[PipelineRow] = []
        last_line = start_line
        done = False

        for line_no, record in _read_jsonl(data_file, start_line=start_line):
            last_line = line_no
            stats.processed += 1

            try:
                row = transform(record, min_year=min_year)
            except Exception as exc:
                log.warning("Transform error at line %d: %s", line_no, exc)
                stats.errors += 1
                continue

            if row is None:
                stats.filtered += 1
                continue

            if max_records > 0:
                remaining = max_records - stats.inserted - len(batch)
                if remaining <= 0:
                    done = True
                    break

            batch.append(row)

            if len(batch) >= batch_size:
                self._commit_batch(batch, line_no, stats)
                batch = []
                self._checkpoint.save(last_line)
                log.info(
                    "line=%-8d  inserted=%-8d  filtered=%-6d  errors=%d",
                    last_line, stats.inserted, stats.filtered, stats.errors,
                )
                if max_records > 0 and stats.inserted >= max_records:
                    done = True
                    break

        if batch:
            self._commit_batch(batch, last_line, stats)
            self._checkpoint.save(last_line)

        if done:
            log.info("--max-records limit (%d) reached — stopping early.", max_records)

        log.info("Load complete. %s", stats)
        return stats

    def _commit_batch(self, batch: list[PipelineRow], line_no: int, stats: LoadStats) -> None:
        db_rows = [r.db_row for r in batch]
        contents = [r.embedding_content for r in batch]

        try:
            product_ids = self._product_repo.bulk_insert(db_rows)
            self._conn.commit()
            stats.inserted += len(batch)
        except Exception as exc:
            self._conn.rollback()
            log.error("Batch insert failed near line %d (%d rows): %s", line_no, len(batch), exc)
            stats.errors += len(batch)
            return

        if product_ids and self._embedder is not None:
            try:
                self._embed_repo.upsert_batch(product_ids, contents, self._embedder)
            except Exception as exc:
                self._conn.rollback()
                log.error("Embedding upsert failed near line %d: %s", line_no, exc)
                stats.errors += len(product_ids)
