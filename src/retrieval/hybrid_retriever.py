"""Hybrid retrieval: BM25 + dense, hop nhat bang RRF.

RRF thay vi weighted-sum vi hai thang diem khong cung don vi
(BM25 khong chan tren, cosine trong [-1,1]) — chuan hoa min-max bi
lech nang khi mot nhanh tra ve it ket qua.
"""

from __future__ import annotations

from ..config import get_settings
from ..embeddings.embedder import Embedder
from ..schemas import Question, RetrievedTable
from ..utils.logging import get_logger
from ..vectordb.bm25_store import BM25Store
from ..vectordb.metadata_store import MetadataStore
from ..vectordb.vector_store import VectorStore
from .filters import CandidateFilter
from .query_analyzer import QueryAnalyzer

log = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Store,
        vectors: VectorStore,
        metadata: MetadataStore,
        embedder: Embedder | None = None,
        analyzer: QueryAnalyzer | None = None,
    ):
        cfg = get_settings().retrieval.get("hybrid", {})
        self.rrf_k = int(cfg.get("rrf_k", 60))
        self.candidate_top_k = int(cfg.get("candidate_top_k", 100))

        filt_cfg = get_settings().retrieval.get("hard_filters", {})
        self.bm25 = bm25
        self.vectors = vectors
        self.metadata = metadata
        self.embedder = embedder or Embedder()
        self.analyzer = analyzer or QueryAnalyzer()
        self.filter = CandidateFilter(
            metadata, year_tolerance=int(filt_cfg.get("year_tolerance", 1))
        )

    # ── main ──────────────────────────────────────────────

    def retrieve(self, question: Question, top_k: int | None = None) -> list[RetrievedTable]:
        top_k = top_k or self.candidate_top_k
        allowed = self.filter.build(question)
        query = self.analyzer.expand_query(question)

        bm25_hits = self.bm25.search(query, top_k=top_k, allowed_ids=allowed)

        qvec = self.embedder.encode_one(query, is_query=True)
        dense_hits = self.vectors.search(qvec, top_k=top_k, allowed_ids=allowed)

        fused = self._rrf(bm25_hits, dense_hits)
        return [self._to_table(ref, sc, bm, dn) for ref, sc, bm, dn in fused[:top_k]]

    # ── fusion ────────────────────────────────────────────

    def _rrf(
        self,
        bm25_hits: list[tuple[str, float]],
        dense_hits: list[tuple[str, float]],
    ) -> list[tuple[str, float, float, float]]:
        scores: dict[str, float] = {}
        bm_raw: dict[str, float] = {}
        dn_raw: dict[str, float] = {}

        for rank, (ref, score) in enumerate(bm25_hits, start=1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (self.rrf_k + rank)
            bm_raw[ref] = score

        for rank, (ref, score) in enumerate(dense_hits, start=1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (self.rrf_k + rank)
            dn_raw[ref] = score

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (ref, score, bm_raw.get(ref, 0.0), dn_raw.get(ref, 0.0))
            for ref, score in ordered
        ]

    # ── helper ────────────────────────────────────────────

    def _to_table(
        self, table_ref: str, score: float, bm25: float, dense: float
    ) -> RetrievedTable:
        meta = self.metadata.get(table_ref)
        return RetrievedTable(
            table_ref=table_ref,
            doc_id=meta.doc_id if meta else table_ref.split("|")[0],
            position=meta.position if meta else int(table_ref.split("|")[-1] or 0),
            score=score,
            csv_path=meta.csv_path if meta else "",
            title=meta.title if meta else "",
            section=meta.section if meta else None,
            card=meta.card if meta else "",
            bm25_score=bm25,
            dense_score=dense,
        )
