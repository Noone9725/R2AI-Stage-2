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
        target_device = device or cfg.get("device", "cpu")
        if target_device == "cuda":
            import torch
            if not torch.cuda.is_available():
                target_device = "cpu"
        self.device = target_device
        self.batch_size = batch_size or cfg.get("batch_size", 32)
        self.normalize = cfg.get("normalize", True) if normalize is None else normalize
        self.max_length = max_length or cfg.get("max_length", 512)
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info("Load embedding model %s tren %s (SentenceTransformer)", self.model_id, self.device)
                m = SentenceTransformer(self.model_id, device=self.device)
                m.max_seq_length = self.max_length
                self._model = m
            except Exception as exc:
                log.warning("Khong load duoc qua SentenceTransformer (%s) -> Dung HuggingFace AutoModel truc tiep", exc)
                self._model = self._load_hf_fallback()
        return self._model

    def _load_hf_fallback(self) -> Any:
        import torch
        from transformers import AutoModel, AutoTokenizer

        log.info("Load embedding model %s tren %s (HuggingFace AutoModel)", self.model_id, self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModel.from_pretrained(self.model_id).to(self.device)
        model.eval()

        class _HFWrapper:
            def __init__(self, tok, mod, device, max_len):
                self.tokenizer = tok
                self.model = mod
                self.device = device
                self.max_length = max_len

            def get_sentence_embedding_dimension(self) -> int:
                return self.model.config.hidden_size

            def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True, **kwargs):
                all_vecs = []
                import torch
                from tqdm import tqdm
                iterator = range(0, len(texts), batch_size)
                if show_progress_bar:
                    iterator = tqdm(iterator, desc="Dense Embeddings", unit="batch")
                
                with torch.no_grad():
                    for i in iterator:
                        batch = texts[i: i + batch_size]
                        inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                        out = self.model(**inputs)
                        # CLS token embedding (bge-m3 dense representation)
                        vecs = out.last_hidden_state[:, 0]
                        if normalize_embeddings:
                            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
                        all_vecs.append(vecs.cpu().numpy())
                return np.vstack(all_vecs) if all_vecs else np.zeros((0, self.get_sentence_embedding_dimension()), dtype=np.float32)

        return _HFWrapper(tokenizer, model, self.device, self.max_length)

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
