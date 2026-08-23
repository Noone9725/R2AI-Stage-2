"""Cross-encoder rerank tren top-N ung vien.

Cross-encoder doc ca query lan card cung luc nen bat duoc quan he ma
bi-encoder bo lo, doi lai cham — chi chay tren ~50 ung vien dau.
Neu khong load duoc model, tra ve nguyen thu tu RRF (degrade, khong crash).
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..schemas import Question, RetrievedTable
from ..utils.logging import get_logger

log = get_logger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None, batch_size: int = 32):
        cfg = get_settings().retrieval.get("rerank", {})
        self.model_name = model_name or cfg.get("model", "BAAI/bge-reranker-v2-m3")
        self.enabled = bool(cfg.get("enabled", True))
        self.top_k = int(cfg.get("top_k", 20))
        self.input_k = int(cfg.get("input_k", 50))
        dev = cfg.get("device", "cuda")
        if dev == "cuda":
            import torch
            if not torch.cuda.is_available():
                dev = "cpu"
        self.device = dev
        self.batch_size = batch_size
        self._model: Any | None = None
        self._failed = False

    @property
    def model(self) -> Any | None:
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                log.info("Load reranker %s", self.model_name)
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:  # noqa: BLE001
                log.warning("Khong load duoc reranker (%s) — bo qua buoc rerank", exc)
                self._failed = True
        return self._model

    def rerank(
        self, question: Question, tables: list[RetrievedTable]
    ) -> list[RetrievedTable]:
        if not self.enabled or not tables:
            return tables[: self.top_k]

        model = self.model
        if model is None:
            return tables[: self.top_k]

        head = tables[: self.input_k]
        tail = tables[self.input_k :]

        pairs = [(question.question, t.card or t.title or t.table_ref) for t in head]
        try:
            scores = model.predict(pairs, batch_size=self.batch_size)
        except Exception as exc:  # noqa: BLE001
            log.warning("Rerank loi (%s) — giu thu tu RRF", exc)
            return tables[: self.top_k]

        for table, score in zip(head, scores):
            table.rerank_score = float(score)
            table.score = float(score)

        head.sort(key=lambda t: t.rerank_score, reverse=True)
        return (head + tail)[: self.top_k]
