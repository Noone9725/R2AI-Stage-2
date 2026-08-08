"""Test evaluator: MRR@5 + 10 metric + failure taxonomy — Phase 4.

Khong chay pipeline — chi unit test logi chay benchmark-code.
"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    Failure,
    answer_correct,
    mrr_at_k,
    precision_recall_f2,
    evaluate,
)


# ── precision/recall/f2 ───────────────────────────────────


def test_prf2_basic() -> None:
    p, r, f2 = precision_recall_f2(["A", "B"], ["A", "C"])
    assert p == pytest.approx(0.5)   # 1/2
    assert r == pytest.approx(0.5)   # 1/2
    assert f2 == pytest.approx(0.5)  # 5*0.25/(4*0.5+0.5)=0.5


def test_prf2_gold_rong_prediction_rong() -> None:
    assert precision_recall_f2([], []) == (1.0, 1.0, 1.0)


def test_prf2_gold_rong_prediction_khong_rong() -> None:
    assert precision_recall_f2(["A"], []) == (0.0, 1.0, 0.0)


def test_prf2_khong_prediction_co_gold() -> None:
    assert precision_recall_f2([], ["A"]) == (0.0, 0.0, 0.0)


# ── MRR@5 ─────────────────────────────────────────────────


def test_mrr_rank1() -> None:
    assert mrr_at_k(["gold1", "x"], ["gold1"]) == 1.0


def test_mrr_rank3() -> None:
    assert mrr_at_k(["a", "b", "gold"], ["gold"]) == pytest.approx(1 / 3)


def test_mrr_ngoai_top5_la_0() -> None:
    assert mrr_at_k(["a", "b", "c", "d", "e", "gold"], ["gold"]) == 0.0


def test_mrr_khong_retrieve_la_0() -> None:
    assert mrr_at_k(["a", "b"], ["gold"]) == 0.0


def test_mrr_nhieu_gold_lay_item_dau_tien() -> None:
    assert mrr_at_k(["x", "g2", "g1"], ["g1", "g2"]) == pytest.approx(1 / 2)


def test_mrr_duplicate_prediction_khong_anh_huong() -> None:
    assert mrr_at_k(["g", "g", "g"], ["g"]) == 1.0


def test_mrr_empty_prediction() -> None:
    assert mrr_at_k([], ["g"]) == 0.0


def test_mrr_empty_gold_la_0() -> None:
    assert mrr_at_k(["a"], []) == 0.0


# ── answer tolerance ──────────────────────────────────────


def test_answer_chinh_xac() -> None:
    assert answer_correct(100.0, 100.0) is True


def test_answer_trong_1_percent() -> None:
    assert answer_correct(101.0, 100.0) is True   # 1%
    assert answer_correct(105.0, 100.0) is False  # 5%


def test_answer_gold_zero() -> None:
    assert answer_correct(0.0, 0.0) is True
    assert answer_correct(1e-7, 0.0) is True      # < abs_tol


def test_answer_nan_va_inf() -> None:
    assert answer_correct(float("nan"), 100.0) is False
    assert answer_correct(100.0, float("nan")) is False
    assert answer_correct(float("inf"), 100.0) is False


# ── evaluate — 10 metric aggregate + execution ────────────


def test_evaluate_10_metric_va_execution() -> None:
    preds = [
        {"id": 1, "answer": 100.0, "relevant_docs": ["d1"], "relevant_tables": ["d1|1"],
         "pandas_query": "df[df.item=='x']"},
    ]
    golds = [
        {"id": 1, "answer": 100.0, "relevant_docs": ["d1"], "relevant_tables": ["d1|1"]},
    ]
    r = evaluate(preds, golds)
    assert r.n == 1
    assert r.doc_f2 == pytest.approx(1.0)
    assert r.table_f2 == pytest.approx(1.0)
    assert r.table_mrr5 == 1.0
    assert r.doc_mrr5 == 1.0
    assert r.answer_accuracy == 1.0
    assert r.execution_accuracy == 1.0


def test_eval_engine_error_zero_exec() -> None:
    """Code sinh ra co query (co text) nhung answer sai (executed_ok=False)."""
    preds = [{"id": 1, "answer": 50.0, "relevant_docs": ["d1"], "relevant_tables": ["d1|1"],
              "pandas_query": "result=wrong"}]
    golds = [{"id": 1, "answer": 100.0, "relevant_docs": ["d1"], "relevant_tables": ["d1|1"]}]
    r = evaluate(preds, golds, executed_ok={1: False})
    assert r.answer_accuracy == 0.0
    assert r.execution_accuracy == 0.0


def test_eval_empty_prediction() -> None:
    preds = [{"id": 1, "answer": None, "relevant_docs": [], "relevant_tables": [],
              "pandas_query": ""}]
    golds = [{"id": 1, "answer": 100.0, "relevant_docs": ["d1"], "relevant_tables": ["d1|1"]}]
    r = evaluate(preds, golds)
    assert r.doc_recall == 0.0 and r.table_recall == 0.0
    assert r.table_f2 == 0.0
    assert r.answer_accuracy == 0.0
    assert r.execution_accuracy == 0.0
    assert r.per_question[0]["failure_tags"] and Failure.TABLE_MISS in r.per_question[0]["failure_tags"]


def test_eval_execution_scoring_code_chay_dung_nhung_answer_sai() -> None:
    preds = [{"id": 1, "answer": 150.0, "relevant_docs": ["d1"], "relevant_tables": ["d1|1"],
              "pandas_query": "result=150"}]
    golds = [{"id": 1, "answer": 100.0, "relevant_docs": ["d1"], "relevant_tables": ["d1|1"]}]
    r = evaluate(preds, golds, executed_ok={1: True})
    assert r.execution_accuracy == 0.0, "chay duoc nhung ket qua sai"