"""Kiem tra tinh nhat quan cua artifact corpus va index.

TAI SAO TON TAI: da tung xay ra tinh trang 125.839 CSV nam tren dia nhung
manifest chi con 1.092 dong — pipeline van chay "binh thuong", chi la
retrieval mat 99% kho du lieu. Khong co exception nao, khong co canh bao
nao. Do la kieu loi dat nhat trong mot cuoc thi co gioi han luot nop.

Nguyen tac: FAIL LOUDLY. Sai lech giua cac tang artifact la loi, khong
phai canh bao — vi tiep tuc infer se cho ra diem thap ma khong ai biet ly do.

Chuoi artifact duoc kiem:
    raw .txt -> processed/*.csv -> manifest.jsonl -> bm25.pkl -> vectors.pkl
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from ..utils.io import read_jsonl
from ..utils.logging import get_logger

log = get_logger(__name__)

_CSV_NAME_RE = re.compile(r"^(?P<doc>.+)_table_(?P<pos>\d+)\.csv$")


class CorpusIntegrityError(RuntimeError):
    """Artifact khong nhat quan — khong duoc phep chay inference tiep."""


@dataclass
class IntegrityReport:
    """Ket qua kiem tra. `ok` False = khong duoc chay inference."""

    # corpus
    raw_documents: int = 0
    processed_csv: int = 0
    manifest_entries: int = 0
    unique_docs: int = 0
    unique_tickers: int = 0
    raw_tickers: int = 0
    orphan_csv: int = 0                 # CSV co tren dia, khong co trong manifest
    missing_csv: int = 0                # manifest tro toi CSV khong ton tai
    duplicate_table_ids: int = 0
    # index
    bm25_entries: int | None = None
    dense_entries: int | None = None
    metadata_entries: int | None = None

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    samples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        lines = [
            "=== CORPUS INTEGRITY ===",
            f"  raw documents      : {self.raw_documents}",
            f"  processed CSV      : {self.processed_csv}",
            f"  manifest entries   : {self.manifest_entries}",
            f"  unique docs        : {self.unique_docs}",
            f"  unique tickers     : {self.unique_tickers} (raw co {self.raw_tickers})",
            f"  orphan CSV         : {self.orphan_csv}",
            f"  missing CSV        : {self.missing_csv}",
            f"  duplicate table_id : {self.duplicate_table_ids}",
            "=== INDEX INTEGRITY ===",
            f"  manifest entries   : {self.manifest_entries}",
            f"  BM25 entries       : {self._fmt(self.bm25_entries)}",
            f"  dense entries      : {self._fmt(self.dense_entries)}",
            f"  metadata entries   : {self._fmt(self.metadata_entries)}",
        ]
        for name, items in self.samples.items():
            if items:
                lines.append(f"  vi du {name}: {items[:3]}")
        if self.errors:
            lines.append(f"LOI ({len(self.errors)}):")
            lines += [f"  x {e}" for e in self.errors]
        if self.warnings:
            lines.append(f"CANH BAO ({len(self.warnings)}):")
            lines += [f"  ! {w}" for w in self.warnings]
        if self.ok:
            lines.append("OK — artifact nhat quan.")
        return "\n".join(lines)

    @staticmethod
    def _fmt(value: int | None) -> str:
        return "(chua build)" if value is None else str(value)


def _count_raw(raw_dir: Path) -> tuple[int, int]:
    """(so file .txt, so thu muc ticker) trong kho raw."""
    if not raw_dir.exists():
        return 0, 0
    n_txt = sum(1 for _ in raw_dir.rglob("*_extracted.txt"))
    root = raw_dir / "financial_statements"
    base = root if root.exists() else raw_dir
    n_ticker = sum(1 for p in base.iterdir() if p.is_dir()) if base.exists() else 0
    return n_txt, n_ticker


def check_corpus(
    manifest_path: Path | None = None,
    processed_dir: Path | None = None,
    raw_dir: Path | None = None,
    *,
    min_ticker_ratio: float = 0.9,
    max_orphan_ratio: float = 0.05,
) -> IntegrityReport:
    """Kiem tra manifest <-> CSV <-> raw.

    min_ticker_ratio: manifest phai phu it nhat ti le nay cua so ticker co
      trong raw. Khong hardcode 100 — dataset khac van dung duoc.
    max_orphan_ratio: ti le CSV mo coi toi da truoc khi coi la loi. Vai file
      mo coi (doi tham so extraction) la binh thuong; 99% thi la bug.
    """
    s = get_settings()
    manifest_path = manifest_path or s.paths.index / "manifest.jsonl"
    processed_dir = processed_dir or s.paths.processed
    raw_dir = raw_dir or s.paths.raw

    rep = IntegrityReport()
    rep.raw_documents, rep.raw_tickers = _count_raw(raw_dir)

    csv_on_disk = (
        {p.name for p in processed_dir.glob("*.csv")} if processed_dir.exists() else set()
    )
    rep.processed_csv = len(csv_on_disk)

    if not manifest_path.exists():
        rep.error(
            f"Chua co manifest: {manifest_path}. Chay: python main.py corpus"
        )
        return rep

    rows = list(read_jsonl(manifest_path))
    rep.manifest_entries = len(rows)
    if not rows:
        rep.error("Manifest rong — khong co bang nao de index.")
        return rep

    refs = Counter(str(r.get("table_ref", "")) for r in rows)
    dups = [ref for ref, n in refs.items() if n > 1]
    rep.duplicate_table_ids = len(dups)
    if dups:
        rep.error(f"{len(dups)} table_ref bi trung trong manifest.")
        rep.samples["table_ref trung"] = dups[:5]

    rep.unique_docs = len({str(r.get("doc_id", "")) for r in rows})
    tickers = {r.get("ticker") for r in rows if r.get("ticker")}
    rep.unique_tickers = len(tickers)

    # manifest -> CSV
    in_manifest = {str(r.get("filename", "")) for r in rows if r.get("filename")}
    missing = sorted(in_manifest - csv_on_disk)
    rep.missing_csv = len(missing)
    if missing:
        rep.error(f"{len(missing)} manifest entry tro toi CSV khong ton tai.")
        rep.samples["CSV thieu"] = missing[:5]

    # CSV -> manifest (chinh la bug da xay ra)
    orphan = sorted(csv_on_disk - in_manifest)
    rep.orphan_csv = len(orphan)
    if orphan:
        ratio = len(orphan) / max(rep.processed_csv, 1)
        msg = (
            f"{len(orphan)}/{rep.processed_csv} CSV ({ratio:.1%}) co tren dia "
            f"nhung KHONG co trong manifest — retrieval khong bao gio thay chung."
        )
        if ratio > max_orphan_ratio:
            rep.error(msg + " Chay lai full build: python main.py corpus")
        else:
            rep.warn(msg)
        rep.samples["CSV mo coi"] = orphan[:5]

    # do phu ticker
    if rep.raw_tickers:
        ratio = rep.unique_tickers / rep.raw_tickers
        if ratio < min_ticker_ratio:
            rep.error(
                f"Manifest chi phu {rep.unique_tickers}/{rep.raw_tickers} ticker "
                f"({ratio:.1%} < {min_ticker_ratio:.0%}) — corpus build chua hoan tat."
            )

    return rep


def check_index(
    report: IntegrityReport | None = None,
    index_dir: Path | None = None,
    *,
    require_dense: bool = True,
) -> IntegrityReport:
    """Kiem tra BM25/dense/metadata co khop so luong voi manifest khong."""
    s = get_settings()
    index_dir = index_dir or s.paths.index
    rep = report or check_corpus()
    if rep.manifest_entries == 0:
        return rep

    bm25_path = index_dir / "bm25.pkl"
    vec_path = index_dir / "vectors.pkl"

    if not bm25_path.exists():
        rep.error(f"Chua co BM25 index: {bm25_path}. Chay: python main.py index")
    else:
        from ..utils.io import load_pickle

        try:
            rep.bm25_entries = len(load_pickle(bm25_path).get("ids", []))
        except Exception as exc:  # noqa: BLE001
            rep.error(f"Khong doc duoc BM25 index ({bm25_path}): {exc}")

    if not vec_path.exists():
        msg = f"Chua co dense index: {vec_path}. Chay: python main.py index"
        rep.error(msg) if require_dense else rep.warn(msg)
    else:
        from ..utils.io import load_pickle

        try:
            rep.dense_entries = len(load_pickle(vec_path).get("ids", []))
        except Exception as exc:  # noqa: BLE001
            rep.error(f"Khong doc duoc dense index ({vec_path}): {exc}")

    rep.metadata_entries = rep.manifest_entries

    for name, count in (("BM25", rep.bm25_entries), ("dense", rep.dense_entries)):
        if count is not None and count != rep.manifest_entries:
            rep.error(
                f"{name} index co {count} muc nhung manifest co "
                f"{rep.manifest_entries} — index cu/khong dong bo. "
                "Chay lai: python main.py index"
            )

    return rep


def verify_or_raise(*, require_dense: bool = True) -> IntegrityReport:
    """Kiem tra day du; nem CorpusIntegrityError neu khong dat.

    Dung o dau inference — tha dung ngay con hon chay het 1012 cau roi moi
    phat hien index chi co 1 ticker.
    """
    rep = check_index(check_corpus(), require_dense=require_dense)
    if not rep.ok:
        raise CorpusIntegrityError(rep.render())
    return rep
