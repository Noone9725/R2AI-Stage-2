"""Ghep ket qua thanh SubmissionItem dung spec BTC.

Nguyen tac: KHONG BAO GIO bo sot cau hoi. Cau tra loi sai duoc 0 diem
cho truc Answer, nhung thieu id co the lam bai nop khong hop le. Vi vay
moi duong that bai deu tra ve item hoan chinh voi fallback_answer.
"""

from __future__ import annotations

from ..config import get_settings
from ..normalization.answer_unit import AnswerNormalizer, AskedUnit
from ..schemas import (
    Evidence,
    ExecutionResult,
    GeneratedQuery,
    Question,
    RetrievalResult,
    SubmissionItem,
)
from ..utils.logging import get_logger

log = get_logger(__name__)


class SubmissionBuilder:
    def __init__(self, fallback_answer: float | None = None, csv_prefix: str | None = None):
        cfg = get_settings().submission
        self.fallback = float(
            fallback_answer if fallback_answer is not None
            else cfg.get("fallback_answer", 0.0)
        )
        self.csv_prefix = csv_prefix or cfg.get("csv_prefix", "data/")
        self.round_to = int(cfg.get("round_to", 2))
        # Doi ket qua (luon la VND o noi bo) sang don vi cau hoi yeu cau.
        # Deterministic — dat o day de LLM/self-repair khong the lam truot.
        self.units = AnswerNormalizer(round_to=self.round_to)

    def build(
        self,
        question: Question,
        retrieval: RetrievalResult,
        query: GeneratedQuery,
        execution: ExecutionResult | None,
    ) -> SubmissionItem:
        answer = self.fallback
        if execution is not None and execution.success and execution.value is not None:
            raw_f = float(execution.value)
            # Uu tien gia tri tinh toan truc tiep tu ma pandas da duoc lam tron
            # Neu so qua lon (>= 10^11) va cau hoi hoi ty/trieu thi normalizer ho tro du phong
            asked = self._asked_unit(question)
            if raw_f >= 1e11 and asked in (AskedUnit.TRIEU_DONG, AskedUnit.TY_DONG):
                norm = self.units.apply(raw_f, asked)
                answer = norm.value
            else:
                answer = round(raw_f, self.round_to)
        else:
            log.debug(
                "Q%d dung fallback (%s)",
                question.id, execution.error if execution else "khong thuc thi",
            )

        return SubmissionItem(
            id=int(question.id),
            question=question.question,
            answer=answer,
            relevant_docs=self._dedup(retrieval.doc_ids),
            relevant_tables=self._dedup(retrieval.table_refs),
            evidence=self._evidence(query),
            pandas_query=query.pandas_query or "",
        )

    # ── helpers ───────────────────────────────────────────

    @staticmethod
    def _asked_unit(question: Question) -> AskedUnit:
        """Don vi dap an. Uu tien truong da phan tich; fallback: doc lai text.

        Fallback ton tai vi `AnswerPipeline` co duong tao Question truc tiep
        (nhanh xu ly exception) — khi do `asked_unit` chua duoc dien.
        """
        raw = getattr(question, "asked_unit", None) or "none"
        try:
            asked = AskedUnit(raw)
        except ValueError:
            asked = AskedUnit.NONE
        if asked is AskedUnit.NONE and question.question:
            from ..normalization.answer_unit import detect_asked_unit

            asked = detect_asked_unit(question.question)
        return asked

    def _evidence(self, query: GeneratedQuery) -> list[Evidence]:
        return [
            Evidence(variable=var, csv_path=self._norm_path(path))
            for var, path in query.var_to_csv.items()
        ]

    def _norm_path(self, path: str) -> str:
        p = path.replace("\\", "/").lstrip("./")
        if not p.startswith(self.csv_prefix):
            p = self.csv_prefix + p.split("/")[-1]
        return p

    @staticmethod
    def _dedup(values: list[str]) -> list[str]:
        return list(dict.fromkeys(v for v in values if v))
