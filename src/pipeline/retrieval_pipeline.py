"""Giai doan 1 (Phase 1): Retrieval Pipeline.

Chay doc lap sieu nhanh (~1-2 phut cho 1012 cau) tren CPU/GPU.
Doc questions.jsonl -> QueryAnalyzer + HybridRetriever + TableSelector -> outputs/retrieval/retrieval_results.json.
Dung de danh gia ngay DOCS_F2 va TABLES_F2 hoac dong goi retrieval_submission.zip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import time

from ..config import get_settings
from ..embeddings.embedder import Embedder
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.query_analyzer import QueryAnalyzer
from ..retrieval.reranker import Reranker
from ..retrieval.selector import TableSelector
from ..schemas import Question, RetrievalResult, RetrievedTable, SubmissionItem
from ..utils.io import read_json, read_jsonl, write_json
from ..utils.logging import get_logger
from ..vectordb.bm25_store import BM25Store
from ..vectordb.metadata_store import MetadataStore
from ..vectordb.vector_store import VectorStore
from .integrity import verify_or_raise

log = get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        selector: TableSelector | None = None,
        analyzer: QueryAnalyzer | None = None,
        load_index: bool = True,
    ) -> None:
        self.analyzer = analyzer or QueryAnalyzer()
        self.selector = selector or TableSelector()
        self.reranker = Reranker()
        self.retriever = retriever
        if self.retriever is None and load_index:
            self.load_index()

    def load_index(self) -> None:
        vec_path = get_settings().paths.index / "vectors.pkl"
        has_vectors = vec_path.exists()
        verify_or_raise(require_dense=has_vectors)

        metadata = MetadataStore().load()
        vectors = VectorStore.load() if has_vectors else VectorStore()
        self.retriever = HybridRetriever(
            bm25=BM25Store.load(),
            vectors=vectors,
            metadata=metadata,
            embedder=Embedder(device="cpu"),
            analyzer=self.analyzer,
        )
        log.info("Phase 1: Da nap index voi %d bang", len(metadata))

    def run(
        self,
        questions_path: str | Path | None = None,
        out_path: str | Path | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        settings = get_settings()
        q_path = Path(questions_path) if questions_path else settings.paths.questions
        o_path = Path(out_path) if out_path else settings.paths.outputs / "retrieval" / "retrieval_results.json"
        o_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Bat dau Phase 1 (Retrieval): Doc cau hoi tu %s", q_path)
        t0 = time.perf_counter()

        raw_items = list(read_jsonl(q_path))
        if limit is not None:
            raw_items = raw_items[:limit]

        results: list[dict[str, Any]] = []

        for idx, item in enumerate(raw_items, start=1):
            q_id = int(item["id"])
            q_text = str(item["question"])

            # 1. Phan tich cau hoi
            q = self.analyzer.analyze(q_id, q_text)

            # 2. Truy hoi lai
            candidates = self.retriever.retrieve(q)
            candidates = self.reranker.rerank(q, candidates)

            # 3. Chon loc bang va lien ket chuoi bang (Linked Retrieval)
            selection = self.selector.select(q, candidates)

            # Format item trung gian
            doc_ids = selection.doc_ids
            table_refs = selection.table_refs

            tables_data = [
                {
                    "table_ref": t.table_ref,
                    "doc_id": t.doc_id,
                    "position": t.position,
                    "csv_path": t.csv_path,
                    "title": t.title,
                    "section": t.section,
                    "score": round(float(t.score), 4),
                    "is_continuation": t.is_continuation,
                    "group_id": t.group_id,
                    "parent_table_ref": t.parent_table_ref,
                    "next_table_ref": t.next_table_ref,
                }
                for t in selection.tables
            ]

            results.append({
                "id": q_id,
                "question": q_text,
                "relevant_docs": doc_ids,
                "relevant_tables": table_refs,
                "candidate_tables": tables_data,
                "tickers": q.tickers,
                "years": q.years,
                "metrics": q.metrics,
                "asked_unit": q.asked_unit,
                "requested_period": q.requested_period,
                "needs_derived": q.needs_derived,
            })

            if idx % 100 == 0 or idx == len(raw_items):
                elapsed = time.perf_counter() - t0
                log.info("Phase 1: Da truy hoi [%d/%d] cau (%.2fs)", idx, len(raw_items), elapsed)

        write_json(results, o_path)
        total_time = time.perf_counter() - t0
        log.info(
            "Xong Phase 1: %d cau hoi da duoc xu ly trong %.2fs -> %s",
            len(results), total_time, o_path
        )
        return results
