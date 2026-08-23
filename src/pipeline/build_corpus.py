"""Stage 1: .txt -> bang -> CSV chuan hoa + manifest.

Chay MOT LAN (hoac khi doi tham so extraction). Ket qua la data/processed/*.csv
va data/index/manifest.jsonl — dau vao cho ca indexing va inference.
"""

from pathlib import Path

from ..config import get_settings
from ..embeddings.table_card import TableCardBuilder
from ..extraction.table_builder import TableBuilder
from ..extraction.table_detector import TableDetector
from ..ingestion.loader import CorpusLoader
from ..normalization.csv_writer import CsvWriter
from ..normalization.schema_std import SchemaStandardizer
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger

log = get_logger(__name__)


class CorpusPipeline:
    def __init__(self) -> None:
        self.loader = CorpusLoader()
        self.detector = TableDetector()
        self.builder = TableBuilder()
        self.standardizer = SchemaStandardizer()
        self.cards = TableCardBuilder()
        self.writer = CsvWriter()

    def run(
        self,
        limit: int | None = None,
        min_rows: int | None = None,
        resume: bool = True,
    ) -> dict[str, int]:
        cfg = get_settings().corpus
        min_rows = min_rows if min_rows is not None else int(cfg.get("min_table_rows", 2))
        mode = "upsert" if limit is not None else "full"

        ckpt_file = get_settings().paths.index / ".corpus_done_docs.json"
        done_docs: set[str] = set()
        if resume and ckpt_file.exists():
            try:
                data = read_json(ckpt_file)
                if isinstance(data, list):
                    done_docs = set(data)
                    log.info("Resume corpus: Tim thay checkpoint %d doc da xu ly", len(done_docs))
            except Exception as e:
                log.warning("Khong doc duoc checkpoint corpus (%s) -> chay moi", e)

        stats = {
            "docs": 0,
            "resumed_docs": len(done_docs),
            "tables_detected": 0,
            "tables_written": 0,
            "skipped": 0,
        }

        processed_in_this_run = 0
        for doc in self.loader.iter_documents(limit=limit):
            stats["docs"] += 1
            if resume and doc.doc_id in done_docs:
                continue

            tables = self.detector.detect(doc)
            stats["tables_detected"] += len(tables)

            for table in tables:
                self.builder.build(table)          # dien table.df (wide)
                if table.df is None or len(table.df) < min_rows:
                    stats["skipped"] += 1
                    continue

                long_df = self.standardizer.standardize(
                    table, ticker=doc.ticker, fallback_year=doc.year
                )
                if long_df is None or long_df.empty:
                    stats["skipped"] += 1
                    continue

                card = self.cards.build(
                    doc_id=table.doc_id,
                    position=table.position,
                    title=table.title,
                    section=table.section,
                    ticker=doc.ticker,
                    year=doc.year,
                    report_type=doc.report_type,
                    unit=table.unit,
                    df=long_df,
                )
                self.writer.write(
                    table, long_df,
                    card=card,
                    ticker=doc.ticker,
                    year=doc.year,
                    report_type=doc.report_type,
                )
                stats["tables_written"] += 1

            done_docs.add(doc.doc_id)
            processed_in_this_run += 1

            if processed_in_this_run % 20 == 0:
                log.info(
                    "%d doc moi (%d tong da xong) | %d bang ghi | %d bo qua",
                    processed_in_this_run, len(done_docs), stats["tables_written"], stats["skipped"],
                )
                # Ghi checkpoint doc da xong
                try:
                    write_json(list(done_docs), ckpt_file)
                except Exception:
                    pass

        try:
            write_json(list(done_docs), ckpt_file)
        except Exception:
            pass

        manifest = self.writer.write_manifest(mode=mode)
        log.info("Xong stage 1 (mode=%s): %s | manifest -> %s", mode, stats, manifest)
        return stats
