"""Nhan dien khoi bang trong file .txt OCR.

DINH DANG THUC TE (da kiem chung tren 1973 file cua ViFinQA)
------------------------------------------------------------
Kho cua BTC ma hoa bang duoi dang HTML inline `<table><tr><td>`, KHONG
dung pipe/tab. Do do `detect()` uu tien duong HTML (`html_table.py`); ba
dialect van ban ben duoi chi la FALLBACK cho file khong co the <table>
(do trong mau 120 file: 1 file nhu vay).

  HTML      : "<table><tr><td>Chi tieu</td>...</tr></table>"  <- duong chinh
  PIPE      : "| Chi tieu | 2015 | 2014 |"        (markdown/pipe-delimited)
  TAB       : cot ngan cach bang \t
  WHITESPACE: cot ngan cach bang >= 2 dau cach lien tiep

QUAN TRONG: HTML phai duoc phan tich trong van ban THO. `fix_ocr_text`
xoa page-noise theo tung dong va sua token so, co the lam hong markup.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from ..config import get_settings
from ..schemas import ExtractedTable, RawDocument
from ..utils.logging import get_logger
from .html_table import extract_html_tables
from .ocr_fixer import fix_ocr_text, is_numeric_like, is_page_noise

log = get_logger(__name__)


class Dialect(str, Enum):
    HTML = "html"
    PIPE = "pipe"
    TAB = "tab"
    WHITESPACE = "whitespace"


_PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
_WS_SPLIT_RE = re.compile(r"\s{2,}")
_SEPARATOR_ROW_RE = re.compile(r"^[\s\|\-\+=_:]+$")


def _fold(text: str) -> str:
    """Bo dau thanh + ha chu thuong de so khop ben voi loi OCR.

    OCR doi dau thuong xuyen: "BẢNG CẦN ĐỐI KẾ TOẢN" (dung: CÂN ĐỐI KẾ TOÁN).
    So khop co dau se truot het cac bang nay -> mat section/unit.
    Chu 'đ' khong phai dau ket hop nen phai thay tay.
    """
    folded = unicodedata.normalize("NFD", text.lower())
    return "".join(
        c for c in folded if unicodedata.category(c) != "Mn"
    ).replace("đ", "d")

# Tieu de bang trong BCTC Viet Nam. Viet co dau cho de doc; so khop qua
# `_fold` nen loi dau cua OCR khong anh huong.
_TITLE_HINTS = tuple(
    _fold(h)
    for h in (
        "bảng cân đối kế toán", "báo cáo kết quả", "lưu chuyển tiền tệ",
        "tình hình tài chính", "thuyết minh", "chỉ tiêu", "tài sản",
        "nguồn vốn", "tổng cộng", "đơn vị tính",
        "balance sheet", "income statement", "cash flow",
    )
)

# Thu tu QUAN TRONG: "thuyết minh" nam CUOI vi tieu de trang thuyet minh
# ("Thuyet minh BCTC hop nhat (tiep theo)") xuat hien tren rat nhieu bang;
# neu no khop truoc thi bang can doi/KQKD nam trong phan thuyet minh se bi
# gan sai section.
#
# NGAN HANG / CTCK: theo Thong tu 49/2014/TT-NHNN, bang can doi ke toan
# duoc goi la "BAO CAO TINH HINH TAI CHINH" va KQKD la "bao cao ket qua
# hoat dong" (khong co "kinh doanh"). Thieu hai bien the nay thi toan bo
# BCTC ngan hang (ACB, VCB, CTG, SHB, MSB, EIB, VPB, KLB, SSI, MBS...)
# khong co section balance_sheet — do trong mau: 13/80 file.
_SECTION_MAP: tuple[tuple[str, str], ...] = tuple(
    (_fold(needle), label)
    for needle, label in (
        ("cân đối kế toán", "balance_sheet"),
        ("tình hình tài chính", "balance_sheet"),
        ("balance sheet", "balance_sheet"),
        ("kết quả hoạt động", "income_statement"),
        ("kết quả kinh doanh", "income_statement"),
        ("income statement", "income_statement"),
        ("lưu chuyển tiền tệ", "cash_flow"),
        ("cash flow", "cash_flow"),
        ("thuyết minh", "notes"),
    )
)


@dataclass(slots=True)
class _Block:
    start: int
    end: int
    lines: list[str]
    dialect: Dialect


class TableDetector:
    """Cat RawDocument thanh danh sach ExtractedTable (chua co DataFrame)."""

    def __init__(
        self,
        min_rows: int | None = None,
        min_cols: int | None = None,
        context_lines: int = 6,
    ):
        corpus = get_settings().corpus
        self.min_rows = min_rows if min_rows is not None else corpus.get("min_table_rows", 2)
        self.min_cols = min_cols if min_cols is not None else corpus.get("min_table_cols", 2)
        self.context_lines = context_lines

    # ── public API ────────────────────────────────────────

    def detect(self, doc: RawDocument) -> list[ExtractedTable]:
        """Uu tien HTML; chi fallback sang pipe/tab/whitespace khi khong co
        the <table> nao giai ma duoc."""
        html = self._detect_html(doc)
        if html:
            log.debug("%s: %d bang (dialect=html)", doc.doc_id, len(html))
            return html
        return self._detect_text(doc)

    # ── duong HTML (chinh) ────────────────────────────────

    def _detect_html(self, doc: RawDocument) -> list[ExtractedTable]:
        """Giai ma tren van ban THO — truoc khi fix_ocr_text lam hong markup.

        `position` lay tu html_table (dem MOI the <table> ke ca bang bi loc),
        nen no la vi tri that cua bang trong bao cao — dung dinh dang
        `relevant_tables` = "<doc_id>|<position>".
        """
        tables = extract_html_tables(
            doc.text, min_rows=self.min_rows, min_cols=self.min_cols
        )
        out: list[ExtractedTable] = []
        for tb in tables:
            ctx = tb.context_before
            section = self._detect_section(ctx)
            # Neu la bang noi tiep va chua co section, ke thua tu bang truoc
            if tb.is_continuation and not section and out:
                section = out[-1].section
            title = self._title_from_context(ctx)
            if tb.is_continuation and not title and out:
                title = out[-1].title

            out.append(
                ExtractedTable(
                    doc_id=doc.doc_id,
                    position=tb.position,
                    title=title,
                    rows=tb.rows,
                    page=tb.page,
                    section=section,
                    context_before=ctx,
                    is_continuation=tb.is_continuation,
                    group_id=tb.group_id,
                    parent_table_ref=f"{doc.doc_id}|{tb.parent_position}" if tb.parent_position else None,
                    next_table_ref=f"{doc.doc_id}|{tb.next_position}" if tb.next_position else None,
                )
            )
        return out

    @staticmethod
    def _title_from_context(ctx: str) -> str:
        """Dong gan bang nhat co tu khoa BCTC; neu khong co, dong cuoi.

        Bo qua dong "Don vi tinh: VND" khi con lua chon khac — no la chu thich
        don vi, khong phai tieu de bang.
        """
        lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]
        best = ""
        for line in reversed(lines):
            low = _fold(line)
            if not any(hint in low for hint in _TITLE_HINTS):
                continue
            if "don vi tinh" in low:
                best = best or line
                continue
            return line
        return best or (lines[-1] if lines else "")

    # ── duong van ban (fallback) ──────────────────────────

    def _detect_text(self, doc: RawDocument) -> list[ExtractedTable]:
        lines = fix_ocr_text(doc.text).splitlines()
        dialect = self._pick_dialect(lines)

        tables: list[ExtractedTable] = []
        for position, block in enumerate(self._find_blocks(lines, dialect), start=1):
            rows = [
                self._split_row(ln, block.dialect)
                for ln in block.lines
                if not _SEPARATOR_ROW_RE.match(ln.strip())
            ]
            rows = [r for r in rows if any(cell.strip() for cell in r)]
            if len(rows) < self.min_rows:
                continue

            ctx = self._context(lines, block.start)
            tables.append(
                ExtractedTable(
                    doc_id=doc.doc_id,
                    position=position,
                    title=self._find_title(lines, block.start),
                    rows=rows,
                    line_start=block.start,
                    line_end=block.end,
                    section=self._detect_section(ctx),
                    context_before=ctx,
                )
            )

        log.debug("%s: %d bang (dialect=%s)", doc.doc_id, len(tables), dialect.value)
        return tables

    # ── dialect ───────────────────────────────────────────

    def _pick_dialect(self, lines: list[str]) -> Dialect:
        votes: Counter[Dialect] = Counter()
        for line in lines:
            if not self._looks_tabular(line):
                continue
            if line.count("|") >= 2:
                votes[Dialect.PIPE] += 1
            elif line.count("\t") >= 1:
                votes[Dialect.TAB] += 1
            elif len(_WS_SPLIT_RE.split(line.strip())) >= self.min_cols:
                votes[Dialect.WHITESPACE] += 1
        if not votes:
            return Dialect.WHITESPACE
        return votes.most_common(1)[0][0]

    def _looks_tabular(self, line: str) -> bool:
        """Dong co ve thuoc bang: nhieu cot VA co it nhat 1 o dang so.

        Rieng dong header (toan chu) duoc bat qua _is_header_like o buoc
        gom block, nen o day yeu cau co so la an toan.
        """
        stripped = line.strip()
        if not stripped or is_page_noise(stripped):
            return False
        if _SEPARATOR_ROW_RE.match(stripped):
            return False

        for dialect in (Dialect.PIPE, Dialect.TAB, Dialect.WHITESPACE):
            cells = self._split_row(line, dialect)
            cells = [c for c in cells if c.strip()]
            if len(cells) < self.min_cols:
                continue
            if any(is_numeric_like(c) for c in cells):
                return True
        return False

    def _is_header_like(self, line: str) -> bool:
        """Dong tieu de cot: nhieu cot, khong co so, chua tu khoa BCTC."""
        stripped = line.strip().lower()
        if not stripped or _SEPARATOR_ROW_RE.match(stripped):
            return False
        cells = [
            c for c in self._split_row(line, Dialect.WHITESPACE) if c.strip()
        ]
        if len(cells) < self.min_cols:
            return False
        return any(hint in stripped for hint in _TITLE_HINTS)

    @staticmethod
    def _split_row(line: str, dialect: Dialect) -> list[str]:
        if dialect is Dialect.PIPE:
            return [c.strip() for c in _PIPE_SPLIT_RE.split(line.strip().strip("|"))]
        if dialect is Dialect.TAB:
            return [c.strip() for c in line.split("\t")]
        return [c.strip() for c in _WS_SPLIT_RE.split(line.strip())]

    # ── block segmentation ────────────────────────────────

    def _find_blocks(self, lines: list[str], dialect: Dialect) -> list[_Block]:
        blocks: list[_Block] = []
        buf: list[str] = []
        start = 0
        gap = 0
        max_gap = 2  # cho phep 1-2 dong rac chen giua bang

        for i, line in enumerate(lines):
            tabular = self._looks_tabular(line)
            header = not tabular and self._is_header_like(line)

            if tabular or (header and buf):
                if not buf:
                    start = i
                buf.append(line)
                gap = 0
            elif header and not buf:
                start = i
                buf.append(line)
                gap = 0
            elif buf:
                gap += 1
                if gap > max_gap:
                    blocks.append(_Block(start, i - gap, list(buf), dialect))
                    buf.clear()
                    gap = 0
                else:
                    buf.append(line)

        if buf:
            blocks.append(_Block(start, len(lines) - 1, list(buf), dialect))

        return [b for b in blocks if len(b.lines) >= self.min_rows]

    # ── context ───────────────────────────────────────────

    def _context(self, lines: list[str], start: int) -> str:
        lo = max(0, start - self.context_lines)
        return "\n".join(lines[lo:start]).strip()

    def _find_title(self, lines: list[str], start: int) -> str:
        """Dong khong rong gan nhat phia tren co chua tu khoa BCTC."""
        lo = max(0, start - self.context_lines)
        for line in reversed(lines[lo:start]):
            stripped = line.strip()
            if not stripped:
                continue
            low = _fold(stripped)
            if any(hint in low for hint in _TITLE_HINTS):
                return stripped
        # Fallback: dong khong rong dau tien phia tren
        for line in reversed(lines[lo:start]):
            if line.strip():
                return line.strip()
        return ""

    @staticmethod
    def _detect_section(context: str) -> str | None:
        low = _fold(context)
        for needle, label in _SECTION_MAP:
            if needle in low:
                return label
        return None
