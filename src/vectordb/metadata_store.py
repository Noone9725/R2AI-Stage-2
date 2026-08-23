"""Metadata store: table_ref -> thong tin bang.

Doc tu manifest.jsonl. Giu trong DataFrame de loc cung (ticker/year) nhanh
truoc khi rank — day la bo loc quan trong nhat cho precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import get_settings
from ..utils.io import read_jsonl
from ..utils.logging import get_logger

log = get_logger(__name__)

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_TICKER_RE = re.compile(r"^([A-Z0-9]{3,5})_")


@dataclass(slots=True)
class TableMeta:
    table_ref: str
    doc_id: str
    position: int
    csv_path: str
    title: str = ""
    section: str | None = None
    unit: str | None = None
    ticker: str | None = None
    year: int | None = None
    report_type: str | None = None
    card: str = ""
    columns: list[str] | None = None
    n_rows: int = 0


class MetadataStore:
    def __init__(self, path: str | Path | None = None):
        settings = get_settings()
        self.path = Path(path) if path else settings.paths.index / "manifest.jsonl"
        self._df: pd.DataFrame | None = None
        self._by_ref: dict[str, TableMeta] = {}

    # ── load ──────────────────────────────────────────────

    def rebuild_manifest_from_processed(self, processed_dir: str | Path | None = None) -> int:
        """Tu dong scan toan bo CSVs trong data/processed de build lai manifest.jsonl."""
        settings = get_settings()
        p_dir = Path(processed_dir) if processed_dir else settings.paths.processed
        if not p_dir.exists():
            raise FileNotFoundError(f"Khong tim thay thu muc processed: {p_dir}")

        csv_files = sorted(p_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Thu muc {p_dir} khong co file CSV nao! Hay chay stage 1 corpus truoc.")

        log.info("Dang quet %d file CSV trong %s de tao manifest.jsonl...", len(csv_files), p_dir)
        rows: list[dict] = []
        pattern = re.compile(r"^(.*)_table_(\d+)\.csv$")

        for f in csv_files:
            m = pattern.match(f.name)
            if not m:
                continue
            doc_id = m.group(1)
            pos = int(m.group(2))
            table_ref = f"{doc_id}|{pos}"
            ticker = self._infer_ticker(doc_id)
            year = self._infer_year(doc_id)
            report_type = self._infer_report_type(doc_id)
            csv_rel = f"data/{f.name}"

            # Doc nhanh schema va danh sach chi tieu cua file CSV
            items: list[str] = []
            cols: list[str] = []
            try:
                df_sample = pd.read_csv(f, nrows=30)
                cols = list(df_sample.columns)
                if "item" in df_sample.columns:
                    items = [
                        str(x).strip()
                        for x in df_sample["item"].dropna().unique()
                        if str(x).strip() and len(str(x).strip()) > 2
                    ][:15]
            except Exception:
                pass

            # Tao the card co ban giau tin hieu tim kiem
            items_desc = f". Chi tiêu: {', '.join(items)}" if items else ""
            card = f"Bang so {pos} trong BCTC {ticker or ''} nam {year or ''} ({report_type or ''}){items_desc}. Cot: {', '.join(cols[:8])}"

            rows.append({
                "table_ref": table_ref,
                "doc_id": doc_id,
                "position": pos,
                "csv_path": csv_rel,
                "filename": f.name,
                "title": "",
                "section": None,
                "unit": "vnd",
                "ticker": ticker,
                "year": year,
                "report_type": report_type,
                "card": card,
                "columns": cols,
                "n_rows": 0,
            })

        from ..utils.io import write_jsonl_atomic
        self.path.parent.mkdir(parents=True, exist_ok=True)
        n = write_jsonl_atomic(rows, self.path)
        log.info("Da tao thanh cong manifest.jsonl voi %d bang -> %s", n, self.path)
        return n

    def load(self) -> "MetadataStore":
        if not self.path.exists() or self.path.stat().st_size == 0:
            log.info("Chua co manifest.jsonl -> Tu dong khoi tao tu data/processed/...")
            self.rebuild_manifest_from_processed()

        rows = list(read_jsonl(self.path))
        for row in rows:
            # Luon tinh toan chinh xac ticker va report_type tu doc_id
            doc_id = str(row.get("doc_id", ""))
            row["ticker"] = self._infer_ticker(doc_id) or row.get("ticker")
            row["year"] = self._infer_year(doc_id) or row.get("year")
            row["report_type"] = self._infer_report_type(doc_id)

        self._df = pd.DataFrame(rows)
        if "year" in self._df.columns:
            self._df["year"] = pd.to_numeric(self._df["year"], errors="coerce").astype("Int64")

        self._by_ref = {
            row["table_ref"]: TableMeta(
                table_ref=row["table_ref"],
                doc_id=row["doc_id"],
                position=int(row["position"]),
                csv_path=row["csv_path"],
                title=row.get("title", "") or "",
                section=row.get("section"),
                unit=row.get("unit"),
                ticker=row.get("ticker"),
                year=int(row["year"]) if pd.notna(row.get("year")) else None,
                report_type=row.get("report_type"),
                card=row.get("card", "") or "",
                columns=row.get("columns"),
                n_rows=int(row.get("n_rows", 0)),
            )
            for row in rows
        }
        log.info("Load %d bang tu manifest", len(self._by_ref))
        return self

    # ── access ────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df  # type: ignore[return-value]

    def get(self, table_ref: str) -> TableMeta | None:
        if not self._by_ref:
            self.load()
        return self._by_ref.get(table_ref)

    def all_refs(self) -> list[str]:
        if not self._by_ref:
            self.load()
        return list(self._by_ref)

    def __len__(self) -> int:
        if not self._by_ref:
            self.load()
        return len(self._by_ref)

    # ── hard filter ───────────────────────────────────────

    def filter_refs(
        self,
        *,
        tickers: list[str] | None = None,
        years: list[int] | None = None,
        year_tolerance: int = 1,
        sections: list[str] | None = None,
        report_types: list[str] | None = None,
    ) -> set[str] | None:
        """Tap table_ref thoa dieu kien. None = khong loc gi.

        year_tolerance: BCTC nam N thuong chua ca so lieu nam N-1, nen bang
        thuoc bao cao nam N+1 van co the tra loi cau hoi ve nam N.
        """
        df = self.df
        mask = pd.Series(True, index=df.index)
        applied = False

        if tickers and "ticker" in df.columns:
            upper = {t.upper() for t in tickers}
            mask &= df["ticker"].astype(str).str.upper().isin(upper)
            applied = True

        if years and "year" in df.columns:
            allowed: set[int] = set()
            for y in years:
                allowed.update(range(y, y + year_tolerance + 1))
            mask &= df["year"].isin(allowed)
            applied = True

        if report_types and "report_type" in df.columns:
            lower = {r.lower() for r in report_types}
            mask &= df["report_type"].astype(str).str.lower().isin(lower)
            applied = True

        if sections and "section" in df.columns:
            mask &= df["section"].isin(sections)
            applied = True

        if not applied:
            return None

        refs = set(df.loc[mask, "table_ref"].tolist())
        return refs or None    # loc ra rong -> coi nhu khong loc, tranh mat recall

    # ── inference ─────────────────────────────────────────

    @staticmethod
    def _infer_ticker(doc_id: str) -> str | None:
        m = _TICKER_RE.match(doc_id.upper())
        return m.group(1) if m else None

    @staticmethod
    def _infer_year(doc_id: str) -> int | None:
        found = _YEAR_RE.findall(doc_id)
        return int(found[-1]) if found else None

    @staticmethod
    def _infer_report_type(doc_id: str) -> str:
        low = doc_id.lower()
        if "separate" in low or "rieng" in low or "_me" in low or "cong_ty_me" in low or "parent" in low:
            return "separate"
        return "consolidated"
