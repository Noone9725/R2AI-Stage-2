"""Xu ly text tieng Viet: bo dau, chuan hoa, tokenize cho BM25.

Khong dung underthesea/pyvi de tranh phu thuoc nang — BM25 tren token
da bo dau da du tot cho domain thuat ngu tai chinh.
"""

from __future__ import annotations

import re
import unicodedata

# Cac tu khong mang thong tin phan biet trong cau hoi tai chinh
STOPWORDS: frozenset[str] = frozenset({
    "la", "cua", "va", "cho", "voi", "trong", "tren", "duoi", "den", "tu",
    "bao", "nhieu", "the", "nao", "gi", "co", "khong", "nam", "cong", "ty",
    "hay", "hoac", "mot", "cac", "nhung", "duoc", "bi", "se", "da", "dang",
    "thi", "ma", "nay", "do", "vay", "ra", "vao", "ve", "boi",
})

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^0-9a-z]+")
_DUP_PUNCT_RE = re.compile(r"([^\w\s])\1{2,}")


def strip_accents(text: str) -> str:
    """'Doanh thu thuần' -> 'Doanh thu thuan'. Xu ly ca 'đ'/'Đ'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_text(text: str, *, accents: bool = True, lower: bool = True) -> str:
    """Chuan hoa chung: NFC -> (bo dau) -> lowercase -> gom whitespace."""
    text = unicodedata.normalize("NFC", text)
    if accents:
        text = strip_accents(text)
    if lower:
        text = text.lower()
    return normalize_ws(text)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Tokenizer 'vn_simple' dung cho BM25 (khop configs/retrieval.yaml)."""
    flat = normalize_text(text)
    tokens = [t for t in _NON_WORD_RE.split(flat) if t]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def clean_ocr_line(line: str) -> str:
    """Don rac OCR o muc dong: dedupe dau cham, bo ky tu dieu khien."""
    line = "".join(c for c in line if c == "\t" or c == "\n" or ord(c) >= 32)
    line = _DUP_PUNCT_RE.sub(r"\1\1", line)
    return line.rstrip()


def slugify(text: str, *, max_len: int = 80) -> str:
    """Tao ten file an toan tu title bang."""
    flat = normalize_text(text)
    slug = _NON_WORD_RE.sub("_", flat).strip("_")
    return slug[:max_len] or "untitled"
