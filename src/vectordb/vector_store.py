"""Dense vector index (FAISS). Fallback numpy brute-force neu chua cai faiss.

Corpus vai chuc ngan bang -> brute-force numpy van chay duoc (~ms), nen
fallback khong phai do choi: no cho phep chay pipeline trong moi truong
khong cai duoc faiss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from ..utils.io import load_pickle, save_pickle
from ..utils.logging import get_logger

log = get_logger(__name__)


class VectorStore:
    """Luu (ids, vectors) + tim k lang gieng gan nhat theo inner product."""

    def __init__(self, dim: int | None = None, metric: str | None = None):
        cfg = get_settings().retrieval.get("vectordb", {})
        self.metric = (metric or cfg.get("metric", "ip")).lower()
        self.dim = dim

        self.ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._faiss_index: Any | None = None
        self._id_pos: dict[str, int] = {}

    # ── build ─────────────────────────────────────────────

    def build(self, ids: list[str], vectors: np.ndarray) -> "VectorStore":
        if len(ids) != len(vectors):
            raise ValueError("ids va vectors phai cung do dai")

        self.ids = list(ids)
        self._id_pos = {rid: i for i, rid in enumerate(self.ids)}
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.dim = int(self._vectors.shape[1])

        faiss = self._try_faiss()
        if faiss is not None:
            index = (
                faiss.IndexFlatIP(self.dim) if self.metric == "ip"
                else faiss.IndexFlatL2(self.dim)
            )
            index.add(self._vectors)
            self._faiss_index = index
            log.info("FAISS index: %d vector, dim=%d", len(self.ids), self.dim)
        else:
            log.warning("Khong co faiss — dung numpy brute-force (%d vector)", len(self.ids))

        return self

    # ── search ────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if self._vectors is None or not self.ids:
            return []

        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)

        if allowed_ids is not None:
            # Loc truoc roi moi tinh — nhanh hon va khong can over-fetch
            positions = np.array(
                [self._id_pos[i] for i in allowed_ids if i in self._id_pos], dtype=np.int64
            )
            if positions.size == 0:
                return []
            return self._brute_force(q, positions, top_k)

        if self._faiss_index is not None:
            k = min(top_k, len(self.ids))
            scores, idx = self._faiss_index.search(q, k)
            return [
                (self.ids[int(i)], float(s))
                for s, i in zip(scores[0], idx[0]) if i >= 0
            ]

        return self._brute_force(q, np.arange(len(self.ids)), top_k)

    def _brute_force(
        self, q: np.ndarray, positions: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        assert self._vectors is not None
        subset = self._vectors[positions]

        if self.metric == "ip":
            scores = subset @ q[0]
        else:
            scores = -np.linalg.norm(subset - q[0], axis=1)

        k = min(top_k, len(positions))
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]

        return [(self.ids[int(positions[i])], float(scores[i])) for i in order]

    # ── persist ───────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else get_settings().paths.index / "vectors.pkl"
        save_pickle({
            "ids": self.ids, "vectors": self._vectors,
            "dim": self.dim, "metric": self.metric,
        }, target)
        log.info("Da luu vector store -> %s", target)
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "VectorStore":
        target = Path(path) if path else get_settings().paths.index / "vectors.pkl"
        if not target.exists():
            raise FileNotFoundError(f"Chua co vector index: {target}")

        state = load_pickle(target)
        store = cls(dim=state["dim"], metric=state["metric"])
        store.build(state["ids"], state["vectors"])
        return store

    @staticmethod
    def _try_faiss() -> Any | None:
        try:
            import faiss  # type: ignore
            return faiss
        except ImportError:
            return None

    def __len__(self) -> int:
        return len(self.ids)
