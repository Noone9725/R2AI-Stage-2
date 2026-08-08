"""Metric giong BTC: macro P/R/F2, Answer Acc, Execution Acc — them MRR@5.

SOURCE OF TRUTH: `competition_documents/02.evaluation.md` + `03.`.
Nhac lai cac rule da xac minh:
  - Macro average tinh tren TUNG QUERY roi lay trung binh.
  - F2 = 5PR/(4P+R), recall nang ~4 lan precision.
  - Precision = hit/len(pred); Recall = hit/len(gold).
  - Question KHONG co gold (rong) va prediction rong -> (P,R,F2) = (1,1,1).
    Prediction khong rong khi gold rong -> (0,1,0).
  - Question khong co prediction -> (0,0,0).
  - Answer: relative tolerance DEFAULT_REL_TOL (1%), abs tol, NaN/Inf -> sai.
  - Execution Accuracy = code chay duoc VA cho ket qua dung (02 §3.3).

MRR@5: KHONG co definition trong competition docs. Dung standard Reciprocal
Rank o top-5:
    rr = 1 / rank of FIRST relevant item (rank <= 5); 0 neu khong co.
    1        if gold o rank 1
    1/3      neu gold o rank 3
    0        neu gold khong o top5 / khong retrieved / empty
Nhieu gold: chi tinh item DAU TIEN xuat hien trong top-5.
Duplicate prediction: khong anh huong RR (chi tim item dau tien).

DOCS MRR bang cong thuc do, tren `relevant_docs` (list co thu tu).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

DEFAULT_REL_TOL = 0.01      # 1% — BTC chua cong bo
DEFAULT_ABS_TOL = 1e-6
MRR_K = 5


# ── retrieval ─────────────────────────────────────────────


def precision_recall_f2(
    predicted: Iterable[str], gold: Iterable[str]
) -> tuple[float, float, float]:
    pred, truth = set(predicted), set(gold)

    if not truth:
        # Khong co nhan duong -> tra rong moi la dung
        return (1.0, 1.0, 1.0) if not pred else (0.0, 1.0, 0.0)
    if not pred:
        return 0.0, 0.0, 0.0

    hits = len(pred & truth)
    p = hits / len(pred)
    r = hits / len(truth)
    f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
    return p, r, f2


def mrr_at_k(predicted: Iterable[str], gold: Iterable[str], k: int = MRR_K) -> float:
    """Reciprocal Rank @k tren list prediction co THU TU.

    RR = 1/rank cua gold DAU TIEN xuat hien trong top-k; 0 neu khong co.
    """
    truth = set(gold)
    if not truth:
        return 0.0
    for i, item in enumerate(predicted, start=1):
        if i > k:
            break
        if item in truth:
            return 1.0 / i
    return 0.0


# ── answer ────────────────────────────────────────────────


def answer_correct(
    pred: float | None,
    gold: float | None,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None
    try:
        p, g = float(pred), float(gold)
    except (TypeError, ValueError):
        return False
    if p != p or g != g:                      # NaN
        return False
    if abs(p - g) <= abs_tol:
        return True
    if g == 0:
        return abs(p) <= abs_tol
    return abs(p - g) / abs(g) <= rel_tol


# ── failure taxonomy ──────────────────────────────────────


class Failure:
    """Tag loi, theo dependency chain — gắn vao moi question."""

    DOC_MISS = "DOC_MISS"
    DOC_BAD_RANK = "DOC_BAD_RANK"
    TABLE_MISS = "TABLE_MISS"
    TABLE_BAD_RANK = "TABLE_BAD_RANK"
    WRONG_REPORT_TYPE = "WRONG_REPORT_TYPE"
    WRONG_PERIOD = "WRONG_PERIOD"
    ROW_AMBIGUITY = "ROW_AMBIGUITY"
    COLUMN_AMBIGUITY = "COLUMN_AMBIGUITY"
    WRONG_UNIT = "WRONG_UNIT"
    QUERY_NO_CODE = "QUERY_NO_CODE"
    QUERY_SYNTAX_ERROR = "QUERY_SYNTAX_ERROR"
    QUERY_SCHEMA_ERROR = "QUERY_SCHEMA_ERROR"
    QUERY_ITEM_ERROR = "QUERY_ITEM_ERROR"
    QUERY_YEAR_ERROR = "QUERY_YEAR_ERROR"
    QUERY_OPERATION_ERROR = "QUERY_OPERATION_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    ANSWER_MISMATCH = "ANSWER_MISMATCH"
    SUCCESS = "SUCCESS"


def classify_failures(
    row: dict[str, Any], pred: dict[str, Any], gold: dict[str, Any]
) -> list[str]:
    """Gan tag loi cho mot question. `row` la per-question metrics.

    Quan trong: phan biet UPSTREAM (khong co du lieu dung) voi DOWNSTREAM
    (co du lieu nhung sinh code/loi). Chon ROOT thi o dependency chain.
    """
    tags: list[str] = []
    gold_docs = set(gold.get("relevant_docs", []))
    gold_tables = set(gold.get("relevant_tables", []))
    pred_docs = set(pred.get("relevant_docs", []))
    pred_tables = set(pred.get("relevant_tables", []))

    # LOC theo chuoi dependency: table miss la root truoc het
    if gold_tables and not (pred_tables & gold_tables):
        tags.append(Failure.TABLE_MISS)
    if gold_docs and not (pred_docs & gold_docs):
        tags.append(Failure.DOC_MISS)

    # rank: co du lieu nhung khong top-5 -> BAD_RANK (chi khi da hit)
    if not tags:
        if gold_tables and row["table_recall"] > 0 and row["table_mrr5"] < 1.0:
            tags.append(Failure.TABLE_BAD_RANK)
        if gold_docs and row["doc_recall"] > 0 and row["doc_mrr5"] < 1.0:
            tags.append(Failure.DOC_BAD_RANK)

    # neu answer sai, phan loai sau
    if not row["answer_correct"]:
        code = str(pred.get("pandas_query", "") or "")
        if not code.strip():
            tags.append(Failure.QUERY_NO_CODE)
        elif row["execution_correct"] is False and pred.get("answer") is not None:
            tags.append(Failure.EXECUTION_ERROR)
        elif not tags:
            tags.append(Failure.ANSWER_MISMATCH)

    if not tags and row["answer_correct"]:
        tags.append(Failure.SUCCESS)
    return tags


# ── aggregate ─────────────────────────────────────────────


@dataclass
class EvalReport:
    n: int = 0
    doc_precision: float = 0.0
    doc_recall: float = 0.0
    doc_f2: float = 0.0
    doc_mrr5: float = 0.0
    table_precision: float = 0.0
    table_recall: float = 0.0
    table_f2: float = 0.0
    table_mrr5: float = 0.0
    answer_accuracy: float = 0.0
    execution_accuracy: float = 0.0
    overall: float = 0.0
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join([
            f"So cau danh gia: {self.n}",
            "-- Retrieval (doc) --",
            f"  Precision : {self.doc_precision:.4f}",
            f"  Recall    : {self.doc_recall:.4f}",
            f"  F2        : {self.doc_f2:.4f}",
            f"  MRR@5     : {self.doc_mrr5:.4f}",
            "-- Retrieval (table) --",
            f"  Precision : {self.table_precision:.4f}",
            f"  Recall    : {self.table_recall:.4f}",
            f"  F2        : {self.table_f2:.4f}",
            f"  MRR@5     : {self.table_mrr5:.4f}",
            "-- Answer / Execution --",
            f"  Answer Accuracy    : {self.answer_accuracy:.4f}",
            f"  Execution Accuracy : {self.execution_accuracy:.4f}",
            f"OVERALL (binh quan 3 truc): {self.overall:.4f}",
        ])


def evaluate(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    executed_ok: dict[int, bool] | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
) -> EvalReport:
    """predictions: submission.json da parse. gold: nhan tay cung schema.

    executed_ok: {id: code chay khong loi}. Thieu thi suy tu viec
    prediction co answer khac None hay khong.
    """
    gold_by_id = {int(g["id"]): g for g in gold}
    pred_by_id = {int(p["id"]): p for p in predictions}
    executed_ok = executed_ok or {}

    rows: list[dict[str, Any]] = []
    for qid, g in sorted(gold_by_id.items()):
        p = pred_by_id.get(qid, {})
        pred_docs = p.get("relevant_docs", [])
        pred_tables = p.get("relevant_tables", [])
        gold_docs = g.get("relevant_docs", [])
        gold_tables = g.get("relevant_tables", [])

        dp, dr, df2 = precision_recall_f2(pred_docs, gold_docs)
        tp, tr, tf2 = precision_recall_f2(pred_tables, gold_tables)
        dmrr = mrr_at_k(pred_docs, gold_docs)
        tmrr = mrr_at_k(pred_tables, gold_tables)
        ans_ok = answer_correct(p.get("answer"), g.get("answer"), rel_tol=rel_tol)
        ran_ok = executed_ok.get(qid, bool(str(p.get("pandas_query", "") or "").strip()))

        row = {
            "id": qid,
            "doc_precision": dp, "doc_recall": dr, "doc_f2": df2, "doc_mrr5": dmrr,
            "table_precision": tp, "table_recall": tr, "table_f2": tf2, "table_mrr5": tmrr,
            "answer_correct": ans_ok,
            "execution_correct": bool(ran_ok and ans_ok),
            "execution_status": "ok" if ran_ok else "no_code",
            "pred_answer": p.get("answer"),
            "gold_answer": g.get("answer"),
            "gold_docs": gold_docs,
            "pred_docs": pred_docs,
            "gold_tables": gold_tables,
            "pred_tables": pred_tables,
            "pandas_query": p.get("pandas_query", ""),
            "asking_unit": p.get("asking_unit"),
        }
        row["failure_tags"] = classify_failures(row, p, g)
        rows.append(row)

    n = len(rows)
    if n == 0:
        return EvalReport()

    avg = lambda k: sum(float(r[k]) for r in rows) / n
    report = EvalReport(
        n=n,
        doc_precision=avg("doc_precision"), doc_recall=avg("doc_recall"),
        doc_f2=avg("doc_f2"), doc_mrr5=avg("doc_mrr5"),
        table_precision=avg("table_precision"), table_recall=avg("table_recall"),
        table_f2=avg("table_f2"), table_mrr5=avg("table_mrr5"),
        answer_accuracy=avg("answer_correct"),
        execution_accuracy=avg("execution_correct"),
        per_question=rows,
    )
    report.overall = (
        report.table_f2 + report.answer_accuracy + report.execution_accuracy
    ) / 3
    return report