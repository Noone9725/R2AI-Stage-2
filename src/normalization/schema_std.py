"""Ep bang BCTC ve schema chuan de LLM sinh pandas de doan.

Schema dich (dang LONG — moi dong 1 chi tieu 1 thoi diem):
    ticker | year | period | section | item | item_code
          | metric | value | unit | source_date | table_ref

Vi sao long thay vi wide: cot nam trong BCTC khong nhat quan ("2015",
"Nam 2015", "31/12/2015"). Ep ve long roi loc `df[df.year == 2015]` giup
prompt on dinh, giam loi sinh code.

VI SAO CAN CA `period` (khong chi `year`): BCTC luon co it nhat hai cot
thoi diem — "31/12/N" (so du cuoi ky) va "01/01/N" (so du dau ky). Neu chi
giu con so nam thi ca hai thanh year=N, tao hai dong cung (item, year) voi
gia tri KHAC NHAU. Do tren 3.000 bang: 4.470 collision nhu vay. `period`
phan biet closing/opening/point_in_time/annual; `source_date` giu ngay goc
de truy nguoc. Chi tiet quy uoc: xem `period.py`.
"""

from __future__ import annotations

import re

import pandas as pd

from ..schemas import ExtractedTable
from ..utils.logging import get_logger
from .number_parser import detect_unit, parse_vn_number
from .period import ColumnPeriod, Period, find_period_cols, parse_column_period
from .term_mapper import TermMapper

log = get_logger(__name__)

STANDARD_COLUMNS: tuple[str, ...] = (
    "ticker", "year", "period", "source_date", "section", "item", "item_code",
    "row_idx", "column_label", "metric", "value", "unit", "table_ref",
)

_CODE_RE = re.compile(r"^\d{2,3}$")
_ITEM_HINTS = ("chi tieu", "chỉ tiêu", "khoan muc", "khoản mục", "item", "noi dung")
_CODE_HINTS = ("ma so", "mã số", "code", "ma")
# Cot phu tro — khong phai du lieu, khong phai truc category.
_SKIP_COL_HINTS = ("thuyet minh", "thuyết minh", "ghi chu", "ghi chú", "note", "v.v")


class SchemaStandardizer:
    """ExtractedTable.df (wide, tho) -> DataFrame long chuan hoa."""

    def __init__(self, term_mapper: TermMapper | None = None):
        self.terms = term_mapper or TermMapper()

    def standardize(
        self,
        table: ExtractedTable,
        *,
        ticker: str | None = None,
        fallback_year: int | None = None,
    ) -> pd.DataFrame:
        if table.df is None or table.df.empty:
            return pd.DataFrame(columns=list(STANDARD_COLUMNS))

        df = table.df
        unit, scale = detect_unit(f"{table.title}\n{table.context_before}")
        table.unit, table.unit_scale = unit, scale

        item_col = self._find_item_col(df)
        code_col = self._find_code_col(df, exclude=item_col)
        # (year, period) thay cho year don thuan: '31/12/2015' va '01/01/2015'
        # phai ra hai semantic key khac nhau.
        period_cols = find_period_cols(
            list(df.columns), exclude={item_col, code_col}, context_year=fallback_year
        )
        # Cot du lieu KHONG phai thoi gian -> truc category (bang thuyet minh
        # TSCD: "Nha cua | May moc | Phuong tien..."). Truoc day ca bang nay
        # chi lay MOT o so dau tien, vut bo cac cot con lai; do la 806/2537
        # collision do trong Phase 2.5.
        category_cols = self._find_category_cols(
            df, skip={item_col, code_col} | set(period_cols)
        )

        records: list[dict] = []
        # `row_idx` la vi tri dong trong bang goc — chi de truy nguoc va lam
        # tie-breaker deterministic, KHONG mang ngu nghia tai chinh.
        for row_idx, (_, row) in enumerate(df.iterrows()):
            item = str(row.get(item_col, "")).strip() if item_col else ""
            if not item:
                continue
            item_code = self._extract_code(row, code_col)
            metric = self.terms.map_one(item)

            def add(col: object, value: float, cp: ColumnPeriod | None) -> None:
                records.append(self._record(
                    table, ticker,
                    cp.year if cp else fallback_year,
                    item, item_code, metric, value, unit,
                    period=cp.period.value if cp else Period.UNKNOWN.value,
                    source_date=cp.source_date if cp else None,
                    row_idx=row_idx,
                    column_label=str(col) if col is not None else "",
                ))

            for col, cp in period_cols.items():
                value = parse_vn_number(row.get(col), scale=scale)
                if value is not None:
                    add(col, value, cp)

            for col in category_cols:
                value = parse_vn_number(row.get(col), scale=scale)
                if value is not None:
                    add(col, value, None)

            # Khong nhan dien duoc cot nao -> lay o so dau tien (hanh vi cu)
            if not period_cols and not category_cols:
                value = self._first_numeric(row, skip={item_col, code_col}, scale=scale)
                if value is not None:
                    add(None, value, None)

        out = pd.DataFrame(records, columns=list(STANDARD_COLUMNS))
        if out.empty:
            return out

        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        return out.dropna(subset=["value"]).reset_index(drop=True)

    # ── record ────────────────────────────────────────────

    @staticmethod
    def _record(
        table: ExtractedTable,
        ticker: str | None,
        year: int | None,
        item: str,
        item_code: str | None,
        metric: str | None,
        value: float,
        unit: str | None,
        *,
        period: str = "unknown",
        source_date: str | None = None,
        row_idx: int | None = None,
        column_label: str = "",
    ) -> dict:
        return {
            "ticker": ticker,
            "year": year,
            "period": period,
            "source_date": source_date,
            "section": table.section,
            "item": item,
            "item_code": item_code,
            "row_idx": row_idx,
            "column_label": column_label,
            "metric": metric,
            "value": value,
            "unit": unit or "vnd",
            "table_ref": table.table_ref,
        }

    # ── column role detection ─────────────────────────────

    @staticmethod
    def _find_item_col(df: pd.DataFrame) -> str | None:
        """Cot ten chi tieu: uu tien theo header, sau do theo ty le text."""
        for col in df.columns:
            low = str(col).lower()
            if any(h in low for h in _ITEM_HINTS):
                return col

        best: tuple[float, str] | None = None
        for col in df.columns:
            series = df[col].astype(str).str.strip()
            non_empty = series[series != ""]
            if non_empty.empty:
                continue
            text_ratio = sum(
                1 for v in non_empty if parse_vn_number(v) is None
            ) / len(non_empty)
            avg_len = non_empty.str.len().mean()
            score = text_ratio * min(avg_len / 10.0, 1.0)
            if best is None or score > best[0]:
                best = (score, col)

        return best[1] if best and best[0] > 0.3 else None

    @staticmethod
    def _find_code_col(df: pd.DataFrame, exclude: str | None) -> str | None:
        for col in df.columns:
            if col == exclude:
                continue
            if any(h in str(col).lower() for h in _CODE_HINTS):
                return col
        # Cot ma so: gan nhu toan so 2-3 chu so.
        # KHONG duoc xet cot thoi gian ("31/12/2017", "2017"): cot gia tri
        # nho (100, 250 trieu...) cung khop `_CODE_RE` va se bi nuot lam cot
        # ma so, keo theo mat toan bo so lieu cua bang.
        for col in df.columns:
            if col == exclude or parse_column_period(col) is not None:
                continue
            series = df[col].astype(str).str.strip()
            non_empty = series[series != ""]
            if len(non_empty) < 3:
                continue
            if sum(1 for v in non_empty if _CODE_RE.match(v)) / len(non_empty) > 0.7:
                return col
        return None

    @staticmethod
    def _find_category_cols(
        df: pd.DataFrame, skip: set[object]
    ) -> list[object]:
        """Cot du lieu co truc la CATEGORY, khong phai thoi gian.

        Bang thuyet minh TSCD/hang ton kho rat pho bien co dang:
            Hang muc | Nha cua | May moc | Phuong tien | Thiet bi VP
        Truc cot la LOAI TAI SAN. Truoc day ca bang nay roi vao nhanh
        `_first_numeric` — chi lay MOT o, vut bo phan con lai, va moi dong
        deu mang cung (item, year, period). Do trong Phase 2.5: 806/2537
        collision den tu day.

        Chi nhan cot co du lieu SO that su — cot "Thuyet minh" (so hieu
        note) hay cot rong khong phai du lieu tai chinh.
        """
        out: list[object] = []
        for col in df.columns:
            if col in skip:
                continue
            low = str(col).lower()
            if any(h in low for h in _SKIP_COL_HINTS):
                continue
            series = df[col].astype(str).str.strip()
            non_empty = series[series != ""]
            if non_empty.empty:
                continue
            n_numeric = sum(1 for v in non_empty if parse_vn_number(v) is not None)
            # Da so o phai la so -> day la cot du lieu, khong phai cot text.
            if n_numeric / len(non_empty) >= 0.6:
                out.append(col)
        return out

    @staticmethod
    def _extract_code(row: pd.Series, code_col: str | None) -> str | None:
        if not code_col:
            return None
        val = str(row.get(code_col, "")).strip()
        return val or None

    @staticmethod
    def _first_numeric(
        row: pd.Series, skip: set[str | None], scale: float
    ) -> float | None:
        for col, val in row.items():
            if col in skip:
                continue
            parsed = parse_vn_number(val, scale=scale)
            if parsed is not None:
                return parsed
        return None
