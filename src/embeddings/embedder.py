"""Encode text -> vector. Wrap sentence-transformers, lazy-load model.

Import sentence_transformers/torch chi khi thuc su encode — script chi doc
manifest hoac chay BM25 khong phai tra gia khoi dong CUDA.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import get_settings
from ..utils.logging import get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        max_length: int | None = None,
    ):
        cfg = get_settings().retrieval.get("embedding", {})
        self.model_id = model_id or cfg.get("model_id", "BAAI/bge-m3")
        self.device = device or cfg.get("device", "cpu")
        self.batch_size = batch_size or cfg.get("batch_size", 32)
        self.normalize = cfg.get("normalize", True) if normalize is None else normalize
        self.max_length = max_length or cfg.get("max_length", 512)
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Can sentence-transformers de encode. "
                    "Chay: pip install sentence-transformers"
                ) from exc

            log.info("Load embedding model %s tren %s", self.model_id, self.device)
            self._model = SentenceTransformer(self.model_id, device=self.device)
            self._model.max_seq_length = self.max_length
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """Tra ve (n, dim) float32. Da normalize neu config bat."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prepared = [self._prefix(t, is_query) for t in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str, *, is_query: bool = True) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def _prefix(self, text: str, is_query: bool) -> str:
        """Mot so model (e5) can prefix 'query:'/'passage:'. bge-m3 thi khong."""
        low = self.model_id.lower()
        if "e5" in low:
            return f"{'query' if is_query else 'passage'}: {text}"
        return text
