"""Trich xuat bang HTML inline trong file .txt OCR cua ViFinQA.

DINH DANG THUC TE (da kiem chung tren data/raw)
------------------------------------------------
File OCR giu ranh gioi trang bang `===== PAGE N =====` va ma hoa bang
duoi dang HTML mot dong:

    <table><tr><td>Másố</td><td>TÀI SẢN</td>...</tr><tr>...</tr></table>

Day la dinh dang chinh: mot BCTC dien hinh co ~80 the <table> va gan nhu
KHONG co dong pipe/tab nao. Vi vay `table_detector.py` (pipe/tab/whitespace)
khong dung duoc cho kho nay — module nay moi la duong chinh.

VAN DE ROWSPAN (quan trong)
---------------------------
OCR gop nhieu gia tri vao MOT o khi o do co rowspan:

    <td rowspan="3">963.717.122.052237.314.356.418726.402.765.634</td>

Ba so bi dan lien. May man la quy uoc so Viet Nam nhom 3 chu so, nen
regex greedy `\\d{1,3}(?:\\.\\d{3})*` tach lai duoc DUY NHAT:
    963.717.122.052 | 237.314.356.418 | 726.402.765.634
Ham `split_glued_numbers` lam viec nay; khi so luong tach duoc dung bang
rowspan thi phan phoi tung so cho tung dong, nguoc lai giu nguyen ca o
(tha de nguyen con hon doan sai -> sai so lieu).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(
    r"<t([dh])([^>]*)>(.*?)</t\1>", re.DOTALL | re.IGNORECASE
)
_SPAN_RE = re.compile(r"\browspan\s*=\s*[\"']?(\d+)", re.IGNORECASE)
_COLSPAN_RE = re.compile(r"\bcolspan\s*=\s*[\"']?(\d+)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_PAGE_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.MULTILINE)

# So kieu Viet Nam: nhom nghin bang dau '.', thap phan bang ','.
# Cho phep dau ngoac ke toan. Nhanh `-` don le la o RONG trong BCTC
# (vd "758.600.000.000-" = mot so roi mot o trong bi dan lien).
_VN_NUMBER_RE = re.compile(r"\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?|-")
# O co ve la du lieu so (de phan biet o gop that vs o bi dan so).
_NUMERIC_CELL_RE = re.compile(r"^[\d\s.,()\-]+$")
# Co it nhat mot so lieu tien te (>= 4 chu so co nhom nghin).
_FIGURE_RE = re.compile(r"\d{1,3}(?:\.\d{3})+")


def split_glued_numbers(text: str) -> list[str]:
    """Tach mot chuoi nhieu so Viet Nam bi dan lien.

    Chi tach khi ca chuoi la day so lien tuc (khong con ky tu la), nham
    tranh cat sai o van ban.

    >>> split_glued_numbers("963.717.122.052237.314.356.418")
    ['963.717.122.052', '237.314.356.418']
    """
    stripped = text.strip()
    if not stripped or not _NUMERIC_CELL_RE.match(stripped):
        return [stripped] if stripped else []
    parts: list[str] = []
    pos = 0
    for m in _VN_NUMBER_RE.finditer(stripped):
        if m.start() != pos:  # co ky tu la chen giua -> khong tach
            return [stripped]
        parts.append(m.group())
        pos = m.end()
    if pos != len(stripped) or len(parts) < 2:
        return [stripped]
    return parts


@dataclass(slots=True)
class HtmlTable:
    """Mot the <table> da giai ma thanh luoi o."""

    position: int
    rows: list[list[str]]
    char_start: int
    char_end: int
    page: int | None = None
    context_before: str = ""
    n_cols: int = field(default=0)
    is_continuation: bool = False
    group_id: str | None = None
    parent_position: int | None = None
    next_position: int | None = None

    def __post_init__(self) -> None:
        self.n_cols = max((len(r) for r in self.rows), default=0)


def _clean_cell(raw: str) -> str:
    return unescape(_TAG_RE.sub(" ", raw)).replace("\xa0", " ").strip()


def _parse_rows(inner: str) -> list[list[str]]:
    """Giai ma <tr>/<td> co rowspan + colspan thanh luoi chu nhat."""
    raw_rows = [m.group(1) for m in _ROW_RE.finditer(inner)]
    if not raw_rows:
        return []

    grid: list[list[str | None]] = [[] for _ in raw_rows]
    # carry[col] = (text_con_lai, so_dong_con_lai)
    pending: list[tuple[int, int, list[str]]] = []  # (col, rows_left, values)

    for r, raw in enumerate(raw_rows):
        row = grid[r]
        cells = [
            (m.group(2) or "", _clean_cell(m.group(3)))
            for m in _CELL_RE.finditer(raw)
        ]
        col = 0
        cur: list[tuple[int, int, list[str]]] = []
        for attrs, text in cells:
            # nhuong cho cac o rowspan tu dong tren
            while any(p[0] == col for p in pending):
                p = next(x for x in pending if x[0] == col)
                while len(row) <= col:
                    row.append(None)
                row[col] = p[2].pop(0) if p[2] else ""
                col += 1

            span = int(m.group(1)) if (m := _SPAN_RE.search(attrs)) else 1
            cspan = int(m.group(1)) if (m := _COLSPAN_RE.search(attrs)) else 1

            while len(row) <= col:
                row.append(None)
            if span > 1:
                vals = split_glued_numbers(text)
                if len(vals) == span:
                    # o bi dan nhieu gia tri -> tra ve dung tung dong
                    row[col] = vals[0]
                    cur.append((col, span - 1, vals[1:]))
                elif _FIGURE_RE.search(text):
                    # co so lieu that nhung khong tach duoc dung so dong ->
                    # chi giu o dong dau, KHONG lap (lap se nhan ban so lieu sai)
                    row[col] = text
                    cur.append((col, span - 1, []))
                else:
                    # o gop that (nhan, so hieu thuyet minh) -> lap xuong duoi
                    row[col] = text
                    cur.append((col, span - 1, [text] * (span - 1)))
            else:
                row[col] = text
            col += cspan
            while len(row) < col:
                row.append("")

        # do not cac o rowspan con lai o cuoi dong
        for p in list(pending):
            c = p[0]
            while len(row) <= c:
                row.append(None)
            if row[c] is None:
                row[c] = p[2].pop(0) if p[2] else ""

        pending = [(c, n - 1, v) for c, n, v in pending if n - 1 > 0]
        pending.extend(cur)

    width = max((len(r) for r in grid), default=0)
    return [[("" if c is None else c) for c in r] + [""] * (width - len(r)) for r in grid]


def _page_index(text: str) -> list[tuple[int, int]]:
    """[(char_offset, page_number)] tu cac marker ===== PAGE N =====."""
    return [(m.start(), int(m.group(1))) for m in _PAGE_RE.finditer(text)]


def _page_at(index: list[tuple[int, int]], offset: int) -> int | None:
    page = None
    for start, num in index:
        if start > offset:
            break
        page = num
    return page


def _context_before(text: str, start: int, chars: int) -> str:
    """Van ban thuan truoc bang, dung cho unit/section/title detection.

    Cat tu sau `</table>` gan nhat de khong keo markup cua bang truoc vao,
    roi xoa moi the con lai va cac marker ===== PAGE N =====.
    """
    lo = max(0, start - chars)
    chunk = text[lo:start]
    cut = chunk.rfind("</table>")
    if cut != -1:
        chunk = chunk[cut + len("</table>") :]
    chunk = _PAGE_RE.sub(" ", _TAG_RE.sub(" ", chunk))
    lines = [ln.strip() for ln in chunk.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def link_consecutive_tables(tables: list[HtmlTable], text: str) -> list[HtmlTable]:
    """Lien ket cac bang HTML bi be gay qua ranh gioi trang (Fractured Tables)
    nhung GIU NGUYEN tung bang doc lap theo dung quy cach BTC.
    """
    if len(tables) < 2:
        return tables

    for i in range(len(tables) - 1):
        curr = tables[i]
        nxt = tables[i + 1]
        gap = text[curr.char_end : nxt.char_start]
        gap_clean = _PAGE_RE.sub("", _TAG_RE.sub("", gap)).strip()
        if len(gap_clean) < 30 and nxt.n_cols == curr.n_cols:
            grp = curr.group_id or f"grp_{curr.position}"
            curr.group_id = grp
            nxt.group_id = grp
            nxt.is_continuation = True
            nxt.parent_position = curr.parent_position or curr.position
            curr.next_position = nxt.position
            if not nxt.context_before and curr.context_before:
                nxt.context_before = curr.context_before

    return tables


def extract_html_tables(
    text: str,
    *,
    min_rows: int = 2,
    min_cols: int = 2,
    context_chars: int = 400,
    link_tables: bool = True,
) -> list[HtmlTable]:
    """Tim moi the <table> trong van ban, tra ve theo thu tu xuat hien.

    `position` la vi tri dong bat dau cua the <table> trong file OCR .txt goc
    (1-indexed), dung 100% dung dinh dang `relevant_tables` dang "<doc_id>|<position>".
    """
    pages = _page_index(text)
    out: list[HtmlTable] = []
    for m in _TABLE_RE.finditer(text):
        line_no = text[: m.start()].count("\n") + 1
        rows = _parse_rows(m.group(1))
        if len(rows) < min_rows or max((len(r) for r in rows), default=0) < min_cols:
            continue
        out.append(
            HtmlTable(
                position=line_no,
                rows=rows,
                char_start=m.start(),
                char_end=m.end(),
                page=_page_at(pages, m.start()),
                context_before=_context_before(text, m.start(), context_chars),
            )
        )

    if link_tables and len(out) > 1:
        out = link_consecutive_tables(out, text)

    return out


def count_html_tables(text: str) -> int:
    """Dem the <table> khong giai ma — dung de do dinh dang nhanh."""
    return len(_TABLE_RE.findall(text))

