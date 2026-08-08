"""Client LLM da backend.

Luat thi: chi model open-weight <=14B, phat hanh truoc 2026-06-01.
Ba backend, cung mot interface:
  - vllm        : chay trong process, nhanh nhat khi co GPU
  - transformers: fallback khong can vLLM
  - openai      : tro toi server OpenAI-compatible tu host (vLLM serve / TGI)
                  — van la model open-weight, chi khac cach goi.
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..utils.logging import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, model_id: str | None = None, backend: str | None = None, **kwargs: Any):
        cfg = get_settings().llm
        self.model_id = model_id or cfg.get("model_id", "Qwen/Qwen2.5-14B-Instruct")
        self.backend = (backend or cfg.get("backend", "vllm")).lower()
        self.temperature = float(kwargs.get("temperature", cfg.get("temperature", 0.0)))
        self.top_p = float(kwargs.get("top_p", cfg.get("top_p", 1.0)))
        self.max_tokens = int(kwargs.get("max_tokens", cfg.get("max_tokens", 1024)))
        self.seed = int(kwargs.get("seed", cfg.get("seed", 42)))
        self.dtype = cfg.get("dtype", "auto")
        self.gpu_memory_utilization = float(cfg.get("gpu_memory_utilization", 0.90))
        self.max_model_len = int(cfg.get("max_model_len", 8192))
        self.base_url = cfg.get("base_url", "http://localhost:8000/v1")

        self._engine: Any | None = None
        self._tokenizer: Any | None = None

    # ── public ────────────────────────────────────────────

    def generate(self, prompt: str, system: str | None = None, **overrides: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        if self.backend == "vllm":
            return self._chat_vllm(messages, **overrides)
        if self.backend == "transformers":
            return self._chat_transformers(messages, **overrides)
        if self.backend == "openai":
            return self._chat_openai(messages, **overrides)
        raise ValueError(f"Backend khong ho tro: {self.backend}")

    def generate_batch(self, prompts: list[str], system: str | None = None) -> list[str]:
        """vLLM chay batch nhanh hon nhieu lan goi le. Backend khac lap tuan tu."""
        if self.backend != "vllm":
            return [self.generate(p, system=system) for p in prompts]

        engine, sampling = self._vllm_engine(), self._vllm_sampling()
        texts = [self._apply_template(
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": p}]
        ) for p in prompts]

        outputs = engine.generate(texts, sampling)
        return [o.outputs[0].text.strip() for o in outputs]

    # ── vllm ──────────────────────────────────────────────

    def _vllm_engine(self) -> Any:
        if self._engine is None:
            from vllm import LLM  # type: ignore

            log.info("Khoi tao vLLM: %s", self.model_id)
            self._engine = LLM(
                model=self.model_id,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                seed=self.seed,
                trust_remote_code=True,
            )
        return self._engine

    def _vllm_sampling(self, **overrides: Any) -> Any:
        from vllm import SamplingParams  # type: ignore

        return SamplingParams(
            temperature=overrides.get("temperature", self.temperature),
            top_p=overrides.get("top_p", self.top_p),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            seed=self.seed,
        )

    def _chat_vllm(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        engine = self._vllm_engine()
        text = self._apply_template(messages)
        outputs = engine.generate([text], self._vllm_sampling(**overrides))
        return outputs[0].outputs[0].text.strip()

    def _apply_template(self, messages: list[dict[str, str]]) -> str:
        tok = self.tokenizer
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        return self._tokenizer

    # ── transformers ──────────────────────────────────────

    def _chat_transformers(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        import torch  # type: ignore

        if self._engine is None:
            from transformers import AutoModelForCausalLM  # type: ignore

            log.info("Khoi tao transformers: %s", self.model_id)
            self._engine = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

        tok = self.tokenizer
        text = self._apply_template(messages)
        inputs = tok(text, return_tensors="pt").to(self._engine.device)

        temperature = overrides.get("temperature", self.temperature)
        with torch.no_grad():
            out = self._engine.generate(
                **inputs,
                max_new_tokens=overrides.get("max_tokens", self.max_tokens),
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=self.top_p if temperature > 0 else None,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return tok.decode(gen, skip_special_tokens=True).strip()

    # ── openai-compatible ─────────────────────────────────

    def _chat_openai(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        if self._engine is None:
            from openai import OpenAI  # type: ignore

            self._engine = OpenAI(
                base_url=self.base_url,
                api_key=get_settings().llm_api_key or "EMPTY",
            )

        resp = self._engine.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=overrides.get("temperature", self.temperature),
            top_p=overrides.get("top_p", self.top_p),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            seed=self.seed,
        )
        return (resp.choices[0].message.content or "").strip()
