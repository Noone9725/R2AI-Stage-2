"""Duyet cay thu muc kho BCTC -> RawDocument.

Cau truc THUC TE cua bo du lieu AIGuruTinix/ViFinQA (da kiem chung):
    data/raw/financial_statements/<TICKER>/<YEAR>/<DOC>/<DOC>_extracted.txt

doc_id = <DOC>, tuc ten file BO duoi `.txt` VA bo hau to `_extracted`.
Theo 03.dubmission_instructions.md, BTC vi du duong dan
    ocr_filter\\AAA\\2015\\AAA_financial_statements_2015_consolidated
-> ma bao cao la `AAA_financial_statements_2015_consolidated`. Do la ten
THU MUC tai lieu, khong phai ten file co `_extracted`. Vi vay phai cat
hau to nay, neu khong toan bo `relevant_docs` se sai va mat diem retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..config import get_settings
from ..schemas import RawDocument
from ..utils.io import read_text
from ..utils.logging import get_logger

log = get_logger(__name__)

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_TICKER_RE = re.compile(r"^[A-Z]{3,4}$")

_REPORT_TYPES = {
    "consolidated": "consolidated",
    "hopnhat": "consolidated",
    "separate": "separate",
    "rieng": "separate",
    "parent": "separate",
}


_EXTRACTED_SUFFIX = "_extracted"


def derive_doc_id(path: str | Path) -> str:
    """Ten file bo `.txt` va bo hau to `_extracted`.

    Neu sau khi cat con trung voi ten thu muc cha thi dung luon — day la
    truong hop chuan cua ViFinQA.
    """
    path = Path(path)
    stem = path.stem
    if stem.endswith(_EXTRACTED_SUFFIX):
        stem = stem[: -len(_EXTRACTED_SUFFIX)]
    return stem


class CorpusLoader:
    """Doc kho .txt. Metadata suy ra tu duong dan truoc, ten file sau."""

    def __init__(self, raw_dir: str | Path | None = None, encoding: str | None = None):
        settings = get_settings()
        self.raw_dir = Path(raw_dir) if raw_dir else settings.paths.raw
        self.encoding = encoding or settings.corpus.get("encoding", "utf-8")

    def list_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Khong thay thu muc raw: {self.raw_dir}. "
                "Giai nen kho BCTC cua BTC vao day truoc."
            )
        files = [p for p in sorted(self.raw_dir.rglob("*.txt")) if p.is_file()]
        log.info("Tim thay %d file .txt hop le trong %s", len(files), self.raw_dir)
        return files

    def iter_documents(self, limit: int | None = None) -> Iterator[RawDocument]:
        files = self.list_files()
        yielded = 0
        for path in files:
            if limit is not None and yielded >= limit:
                break
            if not path.is_file() or not path.exists():
                continue
            try:
                doc = self.load_one(path)
                yield doc
                yielded += 1
            except Exception as exc:  # noqa: BLE001 — mot file loi khong duoc chan pipeline
                log.warning("Bo qua %s: %s", path.name, exc)

    def load_one(self, path: str | Path) -> RawDocument:
        path = Path(path)
        text = read_text(path, self.encoding)
        doc_id = derive_doc_id(path)

        ticker = self._infer_ticker(path, doc_id)
        year = self._infer_year(path, doc_id)

        return RawDocument(
            doc_id=doc_id,
            path=str(path),
            text=text,
            ticker=ticker,
            year=year,
            report_type=self._infer_report_type(doc_id),
            n_lines=text.count("\n") + 1,
        )

    # ── metadata inference ────────────────────────────────

    @staticmethod
    def _infer_ticker(path: Path, doc_id: str) -> str | None:
        # Uu tien thu muc cha: ocr_filter/AAA/2015/... -> AAA
        for part in reversed(path.parts[:-1]):
            if _TICKER_RE.match(part):
                return part
        head = doc_id.split("_", 1)[0].upper()
        return head if _TICKER_RE.match(head) else None

    @staticmethod
    def _infer_year(path: Path, doc_id: str) -> int | None:
        for part in reversed(path.parts[:-1]):
            if _YEAR_RE.fullmatch(part):
                return int(part)
        # Ten file thuong co dang <TICKER>_financial_statements_<YEAR>_<type>
        found = _YEAR_RE.findall(doc_id)
        return int(found[-1]) if found else None

    @staticmethod
    def _infer_report_type(doc_id: str) -> str:
        flat = doc_id.lower().replace("_", "").replace("-", "")
        for needle, label in _REPORT_TYPES.items():
            if needle in flat:
                return label
        return "unknown"
