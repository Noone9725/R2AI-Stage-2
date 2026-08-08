"""Stage 2: cau hoi -> retrieve -> sinh pandas -> chay -> SubmissionItem.

Nguyen tac chiu loi: moi cau hoi duoc boc trong try/except rieng. 100 cau
hoi chay 1 lan, mot exception khong duoc phep xoa sach ket qua da co.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..embeddings.embedder import Embedder
from ..execution.self_repair import SelfRepairExecutor
from ..generation.pandas_generator import PandasGenerator
from ..llm.llm_client import LLMClient
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.query_analyzer import QueryAnalyzer
from ..retrieval.reranker import Reranker
from ..retrieval.selector import TableSelector
from ..schemas import GeneratedQuery, Question, SubmissionItem
from ..submission.builder import SubmissionBuilder
from ..utils.logging import get_logger
from ..vectordb.bm25_store import BM25Store
from ..vectordb.metadata_store import MetadataStore
from ..vectordb.vector_store import VectorStore
from .integrity import verify_or_raise

log = get_logger(__name__)


class AnswerPipeline:
    def __init__(self, llm: LLMClient | None = None, load_index: bool = True):
        self.analyzer = QueryAnalyzer()
        self.selector = TableSelector()
        self.reranker = Reranker()
        self.builder = SubmissionBuilder()

        self.llm = llm or LLMClient()
        self.generator = PandasGenerator(llm=self.llm)
        self.executor = SelfRepairExecutor(llm=self.llm)

        self.retriever: HybridRetriever | None = None
        if load_index:
            self.load_index()

    def load_index(self) -> None:
        # Kiem tra artifact TRUOC khi nap: chay 1012 cau tren mot index thieu
        # 99% corpus se ra diem thap ma khong co dau hieu nao bao la sai.
        # `verify_or_raise` neu ro thieu gi va lenh nao can chay.
        verify_or_raise(require_dense=True)

        metadata = MetadataStore().load()
        self.retriever = HybridRetriever(
            bm25=BM25Store.load(),
            vectors=VectorStore.load(),
            metadata=metadata,
            embedder=Embedder(),
            analyzer=self.analyzer,
        )
        log.info("Da nap index: %d bang", len(metadata))

    # ── mot cau hoi ───────────────────────────────────────

    def answer(self, question_id: int, text: str) -> SubmissionItem:
        assert self.retriever is not None, "Chua nap index — goi load_index()"

        question = self.analyzer.analyze(question_id, text)
        candidates = self.retriever.retrieve(question)
        candidates = self.reranker.rerank(question, candidates)
        retrieval = self.selector.select(question, candidates)

        frames, var_to_csv = self.generator.load_frames(retrieval)
        query = self.generator.generate(
            question, retrieval, frames=frames, var_to_csv=var_to_csv
        )

        execution = None
        if query.pandas_query and frames:
            execution, query = self.executor.run(question.question, query, frames)

        return self.builder.build(question, retrieval, query, execution)

    # ── ca bo cau hoi ─────────────────────────────────────

    def run(self, questions: Iterable[dict[str, Any]]) -> list[SubmissionItem]:
        items: list[SubmissionItem] = []
        n_ok = 0

        for raw in questions:
            qid = int(raw["id"])
            text = str(raw.get("question", ""))
            try:
                item = self.answer(qid, text)
            except Exception as exc:  # noqa: BLE001
                log.exception("Q%d loi khong mong doi: %s", qid, exc)
                item = self.builder.build(
                    Question(id=qid, question=text),
                    self._empty_retrieval(qid),
                    GeneratedQuery(pandas_query=""),
                    None,
                )
            items.append(item)
            if item.pandas_query:
                n_ok += 1
            if len(items) % 10 == 0:
                log.info("Da xu ly %d cau (%d co code)", len(items), n_ok)

        log.info("Xong: %d/%d cau co pandas_query", n_ok, len(items))
        return items

    @staticmethod
    def _empty_retrieval(question_id: int) -> Any:
        from ..schemas import RetrievalResult

        return RetrievalResult(question_id=question_id, tables=[])
