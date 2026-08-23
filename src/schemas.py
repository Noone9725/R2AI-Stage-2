"""Kieu du lieu dung chung toan pipeline.

Mot file duy nhat thay cho folder dto/. Nhom theo stage:
  - Ingestion:    RawDocument
  - Extraction:   ExtractedTable
  - Retrieval:    Question, RetrievedTable, RetrievalResult
  - Generation:   GeneratedQuery
  - Execution:    ExecutionResult
  - Submission:   Evidence, SubmissionItem  (khop 1:1 spec BTC)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ──────────────────────────────────────────────────────────
# INGESTION
# ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class RawDocument:
    """Mot file .txt BCTC trong kho du lieu BTC.

    doc_id la ten THU MUC tai lieu (khong phai ten file — file that co them
    hau to `_extracted`), dung y nguyen cho truong `relevant_docs` khi nop.
    Vd: financial_statements/AAA/2015/AAA_financial_statements_2015_consolidated/
            AAA_financial_statements_2015_consolidated_extracted.txt
        -> doc_id = "AAA_financial_statements_2015_consolidated"
    """

    doc_id: str
    path: str
    text: str
    ticker: str | None = None          # AAA, VNM...
    year: int | None = None
    report_type: str | None = None     # consolidated | separate | unknown
    n_lines: int = 0

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()


# ──────────────────────────────────────────────────────────
# EXTRACTION
# ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ExtractedTable:
    """Mot bang so lieu cat ra tu RawDocument.

    `position` la thu tu/vi tri bang trong bao cao theo du lieu BTC cung cap.
    BAT BUOC phai giu xuyen pipeline: `relevant_tables` co dinh dang
    "<doc_id>|<position>". Mat truong nay la mat 1/3 diem.
    """

    doc_id: str
    position: int
    title: str = ""                    # tieu de/caption phia tren bang
    header_rows: list[list[str]] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    line_start: int = 0                # offset dong trong file .txt goc
    line_end: int = 0
    page: int | None = None
    unit: str | None = None            # "VND" | "trieu VND" | "ty VND"
    unit_scale: float = 1.0            # he so nhan ve VND
    section: str | None = None         # balance_sheet | income_statement | cash_flow | notes
    context_before: str = ""           # vai dong truoc bang — giup section/unit detection
    df: pd.DataFrame | None = None     # dien o buoc normalization
    csv_path: str | None = None        # duong dan tuong doi "data/xxx.csv"

    @property
    def table_ref(self) -> str:
        """Chuoi dung cho `relevant_tables`."""
        return f"{self.doc_id}|{self.position}"

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        widths = [len(r) for r in self.rows] or [0]
        return max(widths)


# ──────────────────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class Question:
    """Mot dong trong test set."""

    id: int
    question: str
    # Suy ra tu query_analyzer
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)   # net_revenue, roe...
    needs_derived: bool = False                        # ROE/ROA/tang truong...
    # Don vi cua DAP AN (ty dong / trieu dong / % / lan...). Gia tri noi bo
    # luon la VND; truong nay dung o SubmissionBuilder de doi nguoc.
    asked_unit: str = "none"
    # Thoi diem cau hoi yeu cau: closing | opening | annual | "" (khong ro).
    requested_period: str = ""
    # Loai BCTC: "consolidated" (hop nhat) | "separate" (cong ty me/rieng le) | None
    report_type: str | None = None


@dataclass(slots=True)
class RetrievedTable:
    """Mot bang ung vien kem diem."""

    table_ref: str                     # "<doc_id>|<position>"
    doc_id: str
    position: int
    score: float
    csv_path: str | None = None
    title: str = ""
    section: str | None = None
    card: str = ""                     # text mo ta bang da dung de match
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float | None = None


@dataclass(slots=True)
class RetrievalResult:
    question_id: int
    tables: list[RetrievedTable] = field(default_factory=list)

    @property
    def doc_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.tables:
            seen.setdefault(t.doc_id, None)
        return list(seen)

    @property
    def table_refs(self) -> list[str]:
        return [t.table_ref for t in self.tables]


# ──────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class GeneratedQuery:
    """Cau lenh pandas do LLM sinh, kem mapping bien -> CSV."""

    pandas_query: str
    var_to_csv: dict[str, str] = field(default_factory=dict)  # {"df1": "data/x.csv"}
    reasoning: str = ""
    attempt: int = 0
    raw_response: str = ""


# ──────────────────────────────────────────────────────────
# EXECUTION
# ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    value: float | None = None
    raw_value: Any = None
    error: str = ""
    error_type: str = ""
    code: str = ""
    elapsed_ms: float = 0.0
    attempt: int = 0


# ──────────────────────────────────────────────────────────
# SUBMISSION  — khop 1:1 spec 03.dubmission_instructions.md
# ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class Evidence:
    variable: str                      # ten bien DataFrame dung trong pandas_query
    csv_path: str                      # phai bat dau bang "data/"

    def to_dict(self) -> dict[str, str]:
        return {"variable": self.variable, "csv_path": self.csv_path}


@dataclass(slots=True)
class SubmissionItem:
    id: int
    question: str
    answer: float
    relevant_docs: list[str] = field(default_factory=list)
    relevant_tables: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    pandas_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "question": self.question,
            "answer": float(self.answer),
            "relevant_docs": list(self.relevant_docs),
            "relevant_tables": list(self.relevant_tables),
            "evidence": [e.to_dict() for e in self.evidence],
            "pandas_query": self.pandas_query,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubmissionItem":
        return cls(
            id=int(d["id"]),
            question=str(d.get("question", "")),
            answer=float(d.get("answer", 0.0)),
            relevant_docs=list(d.get("relevant_docs", [])),
            relevant_tables=list(d.get("relevant_tables", [])),
            evidence=[
                Evidence(variable=str(e.get("variable", "")), csv_path=str(e.get("csv_path", "")))
                for e in d.get("evidence", [])
            ],
            pandas_query=str(d.get("pandas_query", "")),
        )
