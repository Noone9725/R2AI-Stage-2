"""Stage 1b: manifest -> BM25 index + dense vector index.

Tach khoi build_corpus vi thuong phai chay lai rieng: doi model embedding
hay doi cach viet card thi khong can extract lai tu .txt (rat lau).
"""

from __future__ import annotations

from ..embeddings.embedder import Embedder
from ..utils.logging import get_logger
from ..vectordb.bm25_store import BM25Store
from ..vectordb.metadata_store import MetadataStore
from ..vectordb.vector_store import VectorStore
from .integrity import CorpusIntegrityError, check_corpus, check_index

log = get_logger(__name__)


class IndexPipeline:
    def __init__(self, embedder: Embedder | None = None):
        self.metadata = MetadataStore()
        self.embedder = embedder or Embedder()

    def run(self, skip_dense: bool = False) -> dict[str, int]:
        # Kiem tra corpus TRUOC khi ton cong embed: index xay tren mot manifest
        # thieu 99% corpus van "thanh cong" ma khong ai biet.
        pre = check_corpus()
        if not pre.ok:
            raise CorpusIntegrityError(pre.render())
        log.info(
            "Corpus OK: %d bang, %d doc, %d ticker",
            pre.manifest_entries, pre.unique_docs, pre.unique_tickers,
        )

        self.metadata.load()
        df = self.metadata.df

        ids = df["table_ref"].astype(str).tolist()
        texts = [
            self._card_text(row)
            for row in df.to_dict("records")
        ]
        log.info("Index %d bang", len(ids))

        bm25 = BM25Store().build(ids, texts)
        bm25.save()

        n_vectors = 0
        if not skip_dense:
            vectors = self.embedder.encode(texts, is_query=False)
            store = VectorStore().build(ids, vectors)
            store.save()
            n_vectors = len(store)

        # Post-build: BM25/dense phai khop so luong voi manifest, neu khong
        # thi da co index cu lan vao — fail ngay, dung de infer chay tiep.
        post = check_index(require_dense=not skip_dense)
        if not post.ok:
            raise CorpusIntegrityError(post.render())
        log.info("Index OK: bm25=%s dense=%s", post.bm25_entries, post.dense_entries)

        return {"tables": len(ids), "bm25": len(bm25.ids), "vectors": n_vectors}

    @staticmethod
    def _card_text(row: dict) -> str:
        """Card la nguon chinh; thieu card thi ghep tu metadata con lai."""
        card = str(row.get("card") or "").strip()
        if card:
            return card
        return " ".join(
            str(row.get(k) or "")
            for k in ("ticker", "year", "section", "title", "unit", "table_ref")
        ).strip()
