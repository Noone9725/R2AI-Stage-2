"""Sua loi OCR pho bien trong BCTC scan.

Chia 2 muc:
  - fix_ocr_text: toan van ban, an toan (khong doi so).
  - fix_numeric_token: chi ap dung khi da xac dinh token LA so — day moi
    la cho duoc phep doi O->0, l->1. Ap dung bua bai len text se pha ten
    chi tieu tieng Viet.
"""

from __future__ import annotations

import re

from ..utils.vn_text import clean_ocr_line

# Ky tu OCR nham thanh dang giong nhau
_LOOKALIKE_DIGITS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "i": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "g": "9",
})

# Cac cum tieng Viet hay bi OCR lam sai trong BCTC
_TERM_FIXES: dict[str, str] = {
    "Doanh thu thuan": "Doanh thu thuần",
    "TONG CONG": "TỔNG CỘNG",
    "TAI SAN": "TÀI SẢN",
    "NGUON VON": "NGUỒN VỐN",
}

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_NUMERIC_SHAPE_RE = re.compile(r"^[\(\-]?[\d\s.,OoIlSsBZzQgD]+[\)%]?$")
_HAS_DIGIT_RE = re.compile(r"\d")
_BLANK_PAGE_NOISE_RE = re.compile(
    r"^\s*(?:trang\s*\d+|page\s*\d+|-\s*\d+\s*-|\d+\s*/\s*\d+)\s*$",
    re.IGNORECASE,
)


def is_numeric_like(token: str) -> bool:
    """Token co 'hinh dang' so ke ca khi con ky tu OCR nham lan.

    Yeu cau it nhat 1 chu so thuc — tranh coi 'Ill' hay 'OOO' la so.
    """
    token = token.strip()
    if not token:
        return False
    return bool(_NUMERIC_SHAPE_RE.match(token)) and bool(_HAS_DIGIT_RE.search(token))


def fix_numeric_token(token: str) -> str:
    """Chi goi khi is_numeric_like(token) da True."""
    return token.translate(_LOOKALIKE_DIGITS)


def is_page_noise(line: str) -> bool:
    return bool(_BLANK_PAGE_NOISE_RE.match(line))


def fix_ocr_text(text: str, *, drop_page_noise: bool = True) -> str:
    """Lam sach toan van ban — KHONG sua chu so o day."""
    out: list[str] = []
    for raw in text.splitlines():
        line = clean_ocr_line(raw)
        if drop_page_noise and is_page_noise(line):
            continue
        for wrong, right in _TERM_FIXES.items():
            if wrong in line:
                line = line.replace(wrong, right)
        out.append(line)
    return "\n".join(out)


def collapse_spaces(line: str, *, keep_columns: bool = True) -> str:
    """Gom whitespace. keep_columns=True giu >=2 space lam ranh gioi cot."""
    if keep_columns:
        return _MULTI_SPACE_RE.sub("  ", line)
    return _MULTI_SPACE_RE.sub(" ", line)
