"""Chon bao nhieu bang de dua vao prompt / nop trong relevant_tables.

F2 = 5PR / (4P + R): recall nang gap 4 lan precision. Voi 1 bang dung:
 - tra 1 bang dung        -> P=1.00 R=1.00 F2=1.00
 - tra 3 bang, 1 dung     -> P=0.33 R=1.00 F2=0.71
 - tra 1 bang, sai        -> P=0    R=0    F2=0
Nen bang thu 2-3 gan nhu luon dang gia: mat ~0.29 F2 khi thua, cuu ca
diem khi bang dau sai. Selector vi vay uu tien BAO PHU hon la sac net.
"""

from __future__ import annotations

from ..config import get_settings
from ..schemas import Question, RetrievalResult, RetrievedTable
from ..utils.logging import get_logger

log = get_logger(__name__)


class TableSelector:
    def __init__(
        self,
        strategy: str | None = None,
        min_tables: int | None = None,
        max_tables: int | None = None,
        score_ratio_threshold: float | None = None,
    ):
        cfg = get_settings().retrieval.get("selector", {})
        self.strategy = strategy or cfg.get("strategy", "adaptive")
        self.min_tables = min_tables if min_tables is not None else int(cfg.get("min_tables", 1))
        self.max_tables = max_tables if max_tables is not None else int(cfg.get("max_tables", 5))
        self.ratio = (
            score_ratio_threshold
            if score_ratio_threshold is not None
            else float(cfg.get("score_ratio_threshold", 0.55))
        )

    def select(
        self, question: Question, tables: list[RetrievedTable]
    ) -> RetrievalResult:
        if not tables:
            return RetrievalResult(question_id=question.id, tables=[])

        if self.strategy == "topk":
            chosen = tables[: self.max_tables]
        else:
            chosen = self._adaptive(question, tables)

        log.debug(
            "Q%d: chon %d/%d bang (%s)",
            question.id, len(chosen), len(tables), self.strategy,
        )
        return RetrievalResult(question_id=question.id, tables=chosen)

    # ── adaptive ──────────────────────────────────────────

    def _adaptive(
        self, question: Question, tables: list[RetrievedTable]
    ) -> list[RetrievedTable]:
        budget = self._budget(question)
        top = tables[0].score
        cutoff = top * self.ratio if top > 0 else float("-inf")

        chosen = [t for t in tables[:budget] if t.score >= cutoff]

        # Dam bao du min_tables va phu het ticker/nam duoc hoi
        if len(chosen) < self.min_tables:
            chosen = tables[: self.min_tables]

        return self._ensure_coverage(question, chosen, tables, budget)

    def _budget(self, question: Question) -> int:
        """Cau hoi nhieu ticker/nam can nhieu bang hon."""
        need = max(1, len(question.tickers)) * max(1, len(question.years))
        if question.needs_derived:
            need += 1          # chi tieu suy dien can >=2 dong tu >=2 bang
        return min(self.max_tables, max(self.min_tables, need))

    def _ensure_coverage(
        self,
        question: Question,
        chosen: list[RetrievedTable],
        pool: list[RetrievedTable],
        budget: int,
    ) -> list[RetrievedTable]:
        """Bo sung bang cho ticker/nam chua duoc bao phu.

        Cau hoi so sanh VNM 2019 vs 2020 ma chi chon bang 2019 thi
        pandas query khong the tinh duoc — recall bang 0.5 la tran.
        """
        if not question.years or len(chosen) >= budget:
            return chosen

        have = {t.doc_id for t in chosen}
        missing = [y for y in question.years if not any(str(y) in d for d in have)]
        if not missing:
            return chosen

        picked = {t.table_ref for t in chosen}
        for year in missing:
            for cand in pool:
                if cand.table_ref in picked:
                    continue
                if str(year) in cand.doc_id or str(year) in cand.card:
                    chosen.append(cand)
                    picked.add(cand.table_ref)
                    break
            if len(chosen) >= self.max_tables:
                break

        return chosen[: self.max_tables]
