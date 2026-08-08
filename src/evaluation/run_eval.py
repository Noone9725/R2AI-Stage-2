"""Chay danh gia local tren nhan tay + in bao cao loi de debug.

Khong co dev set thi khong biet minh dang o dau. File nay cho phep gan
nhan ~50-100 cau roi do lien tuc — vong lap quan trong nhat cua cuoc thi.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from .metrics import EvalReport, evaluate

log = get_logger(__name__)


def run_eval(
    prediction_path: str | Path,
    gold_path: str | Path,
    out_path: str | Path | None = None,
    rel_tol: float = 0.01,
    show_worst: int = 15,
) -> EvalReport:
    preds: list[dict[str, Any]] = read_json(prediction_path)
    gold: list[dict[str, Any]] = read_json(gold_path)

    report = evaluate(preds, gold, rel_tol=rel_tol)
    print(report.render())
    print()
    print(error_breakdown(report))
    print()
    print(funnel(report))
    print()
    print(worst_questions(report, limit=show_worst))

    target = Path(out_path) if out_path else get_settings().paths.outputs / "eval_report.json"
    write_json(
        {
            "summary": {
                k: v for k, v in report.__dict__.items() if k != "per_question"
            },
            "per_question": report.per_question,
        },
        target,
    )
    log.info("Da luu bao cao -> %s", target)
    return report


def error_breakdown(report: EvalReport) -> str:
    """Phan loai cau sai theo NGUYEN NHAN — dung failure taxonomy."""
    counter: Counter[str] = Counter()
    for row in report.per_question:
        for tag in row.get("failure_tags", []):
            counter[tag] += 1

    if not counter:
        return "Khong co cau sai."
    lines = ["Phan loai loi (failure_tags):"]
    lines += [f"  {name:28} {n:5}" for name, n in counter.most_common()]
    return "\n".join(lines)


def funnel(report: EvalReport) -> str:
    """Dep chain: bao nhieu cau qua tung nguong retrieval->exec->answer."""
    doc_hit = sum(1 for r in report.per_question if r["doc_recall"] > 0)
    table_hit = sum(1 for r in report.per_question if r["table_recall"] > 0)
    table_top5 = sum(1 for r in report.per_question if r["table_mrr5"] > 0)
    executed = sum(1 for r in report.per_question if r["execution_status"] == "ok")
    ans_ok = sum(1 for r in report.per_question if r["answer_correct"])

    n_total = report.n
    if not n_total:
        return "RETRIEVAL FUNNEL: (empty)"

    def pct(n: int) -> str:
        return f"{n} ({n / n_total:.1%})"

    lines = [
        "RETRIEVAL FUNNEL (multiplicative):",
        f"  total questions     : {n_total}",
        f"  gold doc retrieved  : {pct(doc_hit)}",
        f"  gold table retrieved: {pct(table_hit)}",
        f"  gold table top-5    : {pct(table_top5)}",
        f"  execution succeeded : {pct(executed)}",
        f"  answer correct      : {pct(ans_ok)}",
    ]
    return "\n".join(lines)


def worst_questions(report: EvalReport, limit: int = 15) -> str:
    bad = [r for r in report.per_question if not r["answer_correct"]]
    bad.sort(key=lambda r: (r["table_f2"], r["id"]))
    if not bad:
        return "Tat ca cau deu dung."

    lines = [f"{min(limit, len(bad))} cau te nhat:"]
    for row in bad[:limit]:
        lines.append(
            f"  id={row['id']:>4}  table_F2={row['table_f2']:.2f}  "
            f"pred={row['pred_answer']}  gold={row['gold_answer']}"
        )
    return "\n".join(lines)
