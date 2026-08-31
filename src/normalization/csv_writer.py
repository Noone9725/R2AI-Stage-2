"""Ghi CSV chuan hoa + manifest.

Rang buoc nop bai: moi `csv_path` trong evidence phai la duong dan tuong
doi bat dau bang "data/". Vi vay ten file duoc sinh o day va ghi thang
vao ExtractedTable.csv_path, roi bundle len submission.

MANIFEST LA SOURCE OF TRUTH cua corpus — index doc tu no, khong scan CSV.
Vi vay `_manifest` chi chua CAC BANG CUA LAN CHAY NAY, va `write_manifest`
phai biet phan biet hai y dinh khac nhau:

    full rebuild  : lan chay quet toan bo dataset -> thay the manifest
    partial run   : lan chay `--limit N` (debug) -> UPSERT vao manifest cu

Truoc day `write_manifest()` luon ghi de. Mot lan chay `--limit 20` da xoa
manifest cua 125.839 CSV, chi con 1.092 dong — CSV van con tren dia nhung
mat hoan toan reference. Xem `mode=` ben duoi.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import get_settings
from ..schemas import ExtractedTable
from ..utils.io import read_jsonl, write_csv, write_jsonl_atomic
from ..utils.logging import get_logger

log = get_logger(__name__)

_UNSAFE_RE = re.compile(r"[^0-9A-Za-z._-]+")

# Khoa duy nhat cua mot bang. `table_ref` = "<doc_id>|<position>" da dinh
# danh duy nhat mot bang trong ca kho: doc_id la ten thu muc tai lieu (duy
# nhat) va position la thu tu the <table> trong tai lieu do. KHONG dung
# filename lam khoa — `_UNSAFE_RE` co the anh xa hai doc_id khac nhau ve
# cung mot ten file.
MANIFEST_KEY = "table_ref"


class CsvWriter:
    """Ghi bang chuan hoa ra data/processed/ + manifest.jsonl."""

    def __init__(self, processed_dir: str | Path | None = None):
        settings = get_settings()
        self.processed_dir = Path(processed_dir) if processed_dir else settings.paths.processed
        self.prefix = settings.submission.get("csv_prefix", "data/")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        # CHI chua cac bang cua lan chay nay — khong phai ca corpus.
        self._manifest: list[dict] = []

    @staticmethod
    def make_filename(doc_id: str, position: int) -> str:
        """`<doc_id>_table_<position>.csv` — khop vi du trong spec BTC."""
        safe = _UNSAFE_RE.sub("_", doc_id).strip("_")
        return f"{safe}_table_{position}.csv"

    def write(
        self,
        table: ExtractedTable,
        df: pd.DataFrame,
        *,
        card: str = "",
        ticker: str | None = None,
        year: int | None = None,
        report_type: str | None = None,
    ) -> str:
        """Ghi mot bang. Tra ve csv_path tuong doi (dung cho evidence)."""
        filename = self.make_filename(table.doc_id, table.position)
        write_csv(df, self.processed_dir / filename)

        csv_path = f"{self.prefix}{filename}"
        table.csv_path = csv_path

        self._manifest.append({
            "table_ref": table.table_ref,
            "doc_id": table.doc_id,
            "position": table.position,
            "csv_path": csv_path,
            "filename": filename,
            "title": table.title,
            "section": table.section,
            "unit": table.unit,
            "ticker": ticker,
            "year": year,
            "report_type": report_type,
            "card": card,
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "columns": [str(c) for c in df.columns],
            "line_start": table.line_start,
            "line_end": table.line_end,
            "is_continuation": table.is_continuation,
            "group_id": table.group_id,
            "parent_table_ref": table.parent_table_ref,
            "next_table_ref": table.next_table_ref,
        })
        return csv_path

    def write_manifest(
        self,
        path: str | Path | None = None,
        *,
        mode: str = "full",
    ) -> Path:
        """Manifest la 'so ho khau' cua corpus — indexing doc tu day.

        mode="full"   : lan chay quet toan dataset. Manifest = ket qua lan
                        chay nay. Dung khi va CHI KHI da xu ly het corpus.
        mode="upsert" : gop vao manifest cu theo `table_ref`. Bang xu ly lai
                        se GHI DE ban cu (khong sinh trung), bang khong dung
                        toi duoc giu nguyen. Day la mode an toan cho `--limit`.

        Ghi atomic: manifest cu chi bi thay khi ban moi da nam tron tren dia.
        """
        target = Path(path) if path else self.processed_dir.parent / "index" / "manifest.jsonl"

        if mode not in ("full", "upsert"):
            raise ValueError(f"mode phai la 'full' hoac 'upsert', nhan: {mode!r}")

        rows = self._merge_rows(target, mode)
        n = write_jsonl_atomic(rows, target)
        log.info("Da ghi manifest (mode=%s): %d bang -> %s", mode, n, target)
        return target

    def _merge_rows(self, target: Path, mode: str) -> list[dict]:
        """Danh sach dong cuoi cung se ghi ra manifest."""
        if mode == "full":
            return list(self._manifest)

        existing: dict[str, dict] = {}
        if target.exists():
            for row in read_jsonl(target):
                key = row.get(MANIFEST_KEY)
                if key:
                    existing[str(key)] = row

        before = len(existing)
        for row in self._manifest:
            key = row.get(MANIFEST_KEY)
            if key:
                existing[str(key)] = row          # upsert: ghi de, khong nhan ban

        log.info(
            "Upsert manifest: %d dong cu + %d dong moi -> %d dong (%d cap nhat)",
            before, len(self._manifest), len(existing),
            before + len(self._manifest) - len(existing),
        )
        return list(existing.values())

    def resolve_local(self, csv_path: str) -> Path:
        """'data/x.csv' -> duong dan that tren dia (de executor doc)."""
        filename = csv_path.replace("\\", "/").split("/")[-1]
        for loc in (
            self.processed_dir / filename,
            self.processed_dir.parent / filename,
            Path("data/processed") / filename,
            Path("data") / filename,
            Path(csv_path),
        ):
            if loc.exists():
                return loc
        return self.processed_dir / filename
