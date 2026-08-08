"""ExtractedTable (rows tho) -> DataFrame co header sach.

Giu nguyen `position` — day la khoa noi ve `relevant_tables` khi nop bai.
"""

from __future__ import annotations

import re

import pandas as pd

from ..schemas import ExtractedTable
from ..utils.logging import get_logger
from .ocr_fixer import is_numeric_like

log = get_logger(__name__)

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_DUP_SUFFIX = "_{}"


class TableBuilder:
    """Xac dinh dong header, chuan hoa ten cot, dung DataFrame dang wide."""

    def __init__(self, max_header_scan: int = 3):
        self.max_header_scan = max_header_scan

    def build(self, table: ExtractedTable) -> ExtractedTable:
        rows = self._pad(table.rows)
        if not rows:
            table.df = pd.DataFrame()
            return table

        header_idx = self._find_header_row(rows)
        if header_idx is None:
            columns = self._synthetic_columns(len(rows[0]))
            body = rows
            table.header_rows = []
        else:
            columns = self._clean_columns(rows[header_idx])
            body = rows[header_idx + 1:]
            table.header_rows = [rows[header_idx]]

        if not body:
            body, columns = rows, self._synthetic_columns(len(rows[0]))
            table.header_rows = []

        body = self._pad(body, width=len(columns))
        table.df = pd.DataFrame(body, columns=columns)
        return table

    # ── header detection ──────────────────────────────────

    def _find_header_row(self, rows: list[list[str]]) -> int | None:
        """Dong header = it o dang so nhat, uu tien co chua nam.

        Trong BCTC header thuong la: ["Chi tieu", "Ma so", "2015", "2014"].
        Chua nam nen khong the dung 'khong co so' lam dieu kien — dung ty le.
        """
        best: tuple[float, int] | None = None
        for i, row in enumerate(rows[: self.max_header_scan]):
            filled = [c for c in row if c.strip()]
            if len(filled) < 2:
                continue
            pure_numeric = sum(
                1 for c in filled if is_numeric_like(c) and not _YEAR_RE.fullmatch(c.strip())
            )
            ratio = pure_numeric / len(filled)
            has_year = any(_YEAR_RE.search(c) for c in filled)
            score = ratio - (0.3 if has_year else 0.0)
            if best is None or score < best[0]:
                best = (score, i)

        if best is None:
            return None
        # Dong toan so thuc su thi khong phai header
        return best[1] if best[0] < 0.6 else None

    # ── column names ──────────────────────────────────────

    def _clean_columns(self, header: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        out: list[str] = []
        for i, cell in enumerate(header):
            name = " ".join(cell.split()).strip() or f"col_{i}"
            if name in seen:
                seen[name] += 1
                name = name + _DUP_SUFFIX.format(seen[name])
            else:
                seen[name] = 0
            out.append(name)
        return out

    @staticmethod
    def _synthetic_columns(width: int) -> list[str]:
        return [f"col_{i}" for i in range(width)]

    # ── shape ─────────────────────────────────────────────

    @staticmethod
    def _pad(rows: list[list[str]], width: int | None = None) -> list[list[str]]:
        """Ep moi dong ve cung so cot (OCR hay thieu/thua o)."""
        if not rows:
            return []
        target = width if width is not None else max(len(r) for r in rows)
        out: list[list[str]] = []
        for row in rows:
            if len(row) < target:
                out.append(list(row) + [""] * (target - len(row)))
            else:
                out.append(list(row[:target]))
        return out
