"""Stage 2: cau hoi -> retrieve -> sinh pandas -> chay -> SubmissionItem.

Nguyen tac chiu loi: moi cau hoi duoc boc trong try/except rieng. 100 cau
hoi chay 1 lan, mot exception khong duoc phep xoa sach ket qua da co.
"""

from __future__ import annotations

from pathlib import Path
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
        from ..config import get_settings
        vec_path = get_settings().paths.index / "vectors.pkl"
        has_vectors = vec_path.exists()
        verify_or_raise(require_dense=has_vectors)

        metadata = MetadataStore().load()
        vectors = VectorStore.load() if has_vectors else VectorStore()
        # Dung CPU cho query embedder (chi 1 cau text ~10ms) de giai phong 100% VRAM GPU cho LLM 14B
        self.retriever = HybridRetriever(
            bm25=BM25Store.load(),
            vectors=vectors,
            metadata=metadata,
            embedder=Embedder(device="cpu"),
            analyzer=self.analyzer,
        )
        log.info("Da nap index: %d bang", len(metadata))

    def answer(self, qid: int, question_text: str) -> SubmissionItem:
        assert self.retriever is not None, "Chua nap index — goi load_index()"
        question = self.analyzer.analyze(qid, question_text)
        log.info("================================================================================")
        log.info("[Q%d] %s", qid, question_text)
        log.info("  * Phan tich: Tickers=%s | Nam=%s | Don vi=%s | Metrics=%s", question.tickers, question.years, question.asked_unit, question.metrics)

        candidates = self.retriever.retrieve(question)
        candidates = self.reranker.rerank(question, candidates)
        retrieval = self.selector.select(question, candidates)
        log.info("  * Bang duoc chon (%d bang): %s", len(retrieval.table_refs), retrieval.table_refs)

        frames, var_to_csv = self.generator.load_frames(retrieval, question=question)
        if not frames:
            log.warning("  * [CANH BAO] Khong load duoc DataFrame nao tu CSV cho Q%d!", qid)
            query = GeneratedQuery(pandas_query="", var_to_csv=var_to_csv, reasoning="No CSV loaded", attempt=0, raw_response="")
            return self.builder.build(question, retrieval, query, None)

        query = self.generator.generate(
            question, retrieval, frames=frames, var_to_csv=var_to_csv
        )

        # Luon goi self.executor de thuc thi hoac tu sua loi / cuu ho ma code
        execution, query = self.executor.run(question.question, query, frames)

        item = self.builder.build(question, retrieval, query, execution)
        return item

    # ── ca bo cau hoi ─────────────────────────────────────

    def run(
        self,
        questions: Iterable[dict[str, Any]],
        *,
        checkpoint_path: Path | str | None = None,
        save_every: int = 5,
        resume: bool = True,
    ) -> list[SubmissionItem]:
        from ..utils.io import read_json, write_json_atomic

        ckpt_file = Path(checkpoint_path) if checkpoint_path else None
        items_by_id: dict[int, SubmissionItem] = {}

        # Resume tu checkpoint neu co
        if ckpt_file and ckpt_file.exists() and resume:
            try:
                existing_data = read_json(ckpt_file)
                if isinstance(existing_data, list):
                    for row in existing_data:
                        qid = int(row["id"])
                        items_by_id[qid] = SubmissionItem.from_dict(row)
                    log.info("Resume tu checkpoint %s: da co %d cau", ckpt_file, len(items_by_id))
            except Exception as exc:  # noqa: BLE001
                log.warning("Khong doc duoc checkpoint (%s) — chay tu dau", exc)

        q_list = list(questions)
        total = len(q_list)
        n_processed = 0

        import time

        def _sync_external_backup(source_file: Path) -> None:
            import shutil
            # 1. Google Drive (Colab)
            drive_root = Path("/content/drive/MyDrive")
            if drive_root.exists():
                backup_dir = drive_root / "backup"
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, backup_dir / source_file.name)
                    log.info("Da backup tu dong sang Google Drive: %s", backup_dir / source_file.name)
                except Exception as e:
                    log.debug("Auto backup Drive: %s", e)

            # 2. Kaggle Working Backup
            kaggle_root = Path("/kaggle/working")
            if kaggle_root.exists():
                backup_dir = kaggle_root / "backup"
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, backup_dir / source_file.name)
                    shutil.copy2(source_file, kaggle_root / source_file.name)
                    log.info("Da backup tu dong sang Kaggle: %s", backup_dir / source_file.name)
                except Exception as e:
                    log.debug("Auto backup Kaggle: %s", e)

        for raw in q_list:
            qid = int(raw["id"])
            if qid in items_by_id:
                continue

            text = str(raw.get("question", ""))
            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0

            items_by_id[qid] = item
            n_processed += 1
            has_query = bool(item.pandas_query and item.pandas_query.strip())
            log.info(
                "[Q%d/%d] Xong trong %.2fs -> answer: %s | bang: %d | query: %s",
                qid, total, elapsed, item.answer, len(item.relevant_tables),
                "OK" if has_query else "EMPTY",
            )

            # Ghi checkpoint theo dot de chong sap giua chung
            if ckpt_file and (n_processed % save_every == 0 or len(items_by_id) == total):
                ordered_items = [
                    items_by_id[int(q["id"])].to_dict()
                    for q in q_list
                    if int(q["id"]) in items_by_id
                ]
                write_json_atomic(ordered_items, ckpt_file)
                log.info("Da luu checkpoint: %d/%d cau -> %s", len(items_by_id), total, ckpt_file)
                _sync_external_backup(ckpt_file)

            if len(items_by_id) % 10 == 0 or len(items_by_id) == total:
                n_ok = sum(1 for it in items_by_id.values() if it.pandas_query)
                log.info("Tien do: %d/%d cau (%d co pandas_query)", len(items_by_id), total, n_ok)

        # Ghi lan cuoi
        if ckpt_file:
            ordered_items = [
                items_by_id[int(q["id"])].to_dict()
                for q in q_list
                if int(q["id"]) in items_by_id
            ]
            write_json_atomic(ordered_items, ckpt_file)
            _sync_external_backup(ckpt_file)

        return [
            items_by_id[int(q["id"])]
            for q in q_list
            if int(q["id"]) in items_by_id
        ]

    def run_from_retrieval(
        self,
        retrieval_path: str | Path,
        *,
        out_path: str | Path | None = None,
        checkpoint_path: Path | str | None = None,
        save_every: int = 5,
        resume: bool = True,
        limit: int | None = None,
    ) -> list[SubmissionItem]:
        """Giai doan 2 (Phase 2): Doc retrieval_results.json -> Sinh ma Pandas -> Thuc thi -> Submission."""
        from ..schemas import RetrievalResult, RetrievedTable
        from ..utils.io import read_json, write_json_atomic

        r_file = Path(retrieval_path)
        if not r_file.exists():
            raise FileNotFoundError(f"Khong tim thay file retrieval trung gian: {r_file}. Hay chay Phase 1 truoc.")

        raw_data = read_json(r_file)
        if not isinstance(raw_data, list):
            raise ValueError(f"File retrieval khong hop le: {r_file} (phai la list JSON)")

        if limit is not None:
            raw_data = raw_data[:limit]

        ckpt_file = Path(checkpoint_path) if checkpoint_path else None
        items_by_id: dict[int, SubmissionItem] = {}

        # Resume tu checkpoint neu co
        if ckpt_file and ckpt_file.exists() and resume:
            try:
                existing_data = read_json(ckpt_file)
                if isinstance(existing_data, list):
                    for row in existing_data:
                        qid = int(row["id"])
                        items_by_id[qid] = SubmissionItem.from_dict(row)
                    log.info("Phase 2: Resume tu checkpoint %s: da co %d cau", ckpt_file, len(items_by_id))
            except Exception as exc:
                log.warning("Phase 2: Khong doc duoc checkpoint (%s) — chay tu dau", exc)

        total = len(raw_data)
        n_processed = 0
        import time

        def _sync_external_backup(source_file: Path) -> None:
            import shutil
            for parent_dir in (Path("/content/drive/MyDrive/backup"), Path("/kaggle/working/backup")):
                if parent_dir.parent.exists():
                    try:
                        parent_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, parent_dir / source_file.name)
                        log.info("Da backup Phase 2 sang: %s", parent_dir / source_file.name)
                    except Exception as e:
                        log.debug("Auto backup: %s", e)

        log.info("Bat dau Phase 2 (Generation & Execution): Xu ly %d cau hoi...", total)

        for item_dict in raw_data:
            qid = int(item_dict["id"])
            if qid in items_by_id:
                continue

            q_text = str(item_dict.get("question", ""))
            question = Question(
                id=qid,
                question=q_text,
                tickers=item_dict.get("tickers", []),
                years=item_dict.get("years", []),
                metrics=item_dict.get("metrics", []),
                asked_unit=item_dict.get("asked_unit", "none"),
                requested_period=item_dict.get("requested_period", ""),
                needs_derived=item_dict.get("needs_derived", False),
            )

            cands = []
            for t in item_dict.get("candidate_tables", []):
                cands.append(
                    RetrievedTable(
                        table_ref=t["table_ref"],
                        doc_id=t["doc_id"],
                        position=int(t["position"]),
                        score=float(t.get("score", 0.0)),
                        csv_path=t.get("csv_path"),
                        title=t.get("title", ""),
                        section=t.get("section"),
                        is_continuation=bool(t.get("is_continuation", False)),
                        group_id=t.get("group_id"),
                        parent_table_ref=t.get("parent_table_ref"),
                        next_table_ref=t.get("next_table_ref"),
                    )
                )
            retrieval = RetrievalResult(question_id=qid, tables=cands)

            t0 = time.perf_counter()
            try:
                frames, var_to_csv = self.generator.load_frames(retrieval, question=question)
                if not frames:
                    log.warning("  * [CANH BAO] Khong load duoc DataFrame cho Q%d!", qid)
                    query = GeneratedQuery(pandas_query="", var_to_csv=var_to_csv, reasoning="No CSV loaded", attempt=0, raw_response="")
                    sub_item = self.builder.build(question, retrieval, query, None)
                else:
                    query = self.generator.generate(
                        question, retrieval, frames=frames, var_to_csv=var_to_csv
                    )
                    execution, query = self.executor.run(question.question, query, frames)
                    sub_item = self.builder.build(question, retrieval, query, execution)
            except Exception as exc:
                log.exception("Q%d Phase 2 loi: %s", qid, exc)
                sub_item = self.builder.build(
                    question, retrieval, GeneratedQuery(pandas_query=""), None
                )

            elapsed = time.perf_counter() - t0
            items_by_id[qid] = sub_item
            n_processed += 1
            has_query = bool(sub_item.pandas_query and sub_item.pandas_query.strip())
            log.info(
                "[Q%d/%d] Phase 2 xong trong %.2fs -> answer: %s | query: %s",
                qid, total, elapsed, sub_item.answer,
                "OK" if has_query else "EMPTY",
            )

            # Checkpoint
            if ckpt_file and (n_processed % save_every == 0 or len(items_by_id) == total):
                ordered = [
                    items_by_id[int(q["id"])].to_dict()
                    for q in raw_data
                    if int(q["id"]) in items_by_id
                ]
                write_json_atomic(ordered, ckpt_file)
                _sync_external_backup(ckpt_file)

        ordered = [
            items_by_id[int(q["id"])]
            for q in raw_data
            if int(q["id"]) in items_by_id
        ]
        if out_path:
            out_file = Path(out_path)
            write_json_atomic([it.to_dict() for it in ordered], out_file)
            log.info("Phase 2 da ghi ket qua cuoi cung -> %s", out_file)
            _sync_external_backup(out_file)

        return ordered

    @staticmethod
    def _empty_retrieval(question_id: int) -> Any:
        from ..schemas import RetrievalResult

        return RetrievalResult(question_id=question_id, tables=[])
