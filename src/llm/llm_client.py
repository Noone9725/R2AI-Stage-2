"""Client LLM da backend.

Luat thi: chi model open-weight <=14B, phat hanh truoc 2026-06-01.
Ba backend, cung mot interface:
  - vllm        : chay trong process, nhanh nhat khi co GPU
  - transformers: fallback khong can vLLM
  - openai      : tro toi server OpenAI-compatible tu host (vLLM serve / TGI)
                  — van la model open-weight, chi khac cach goi.
"""

from __future__ import annotations

import os
from typing import Any

# Su dung vLLM V0 Engine on dinh (tranh deadlock IPC cua V1 engine tren GPU T4)
os.environ.setdefault("VLLM_USE_V1", "0")

from ..config import get_settings
from ..utils.logging import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, model_id: str | None = None, backend: str | None = None, **kwargs: Any):
        cfg = get_settings().llm
        self.model_id = model_id or cfg.get("model_id", "Qwen/Qwen2.5-14B-Instruct")
        self.backend = (backend or cfg.get("backend", "vllm")).lower()
        if self.backend == "vllm":
            try:
                import vllm  # noqa: F401
            except ImportError:
                log.warning("Khong tim thay goi vllm (Windows/moi truong thieu) -> Tu dong fallback sang transformers")
                self.backend = "transformers"
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
        if self._engine is None and self.backend == "vllm":
            from vllm import LLM  # type: ignore

            log.info("Khoi tao vLLM: %s", self.model_id)
            util_candidates = [
                self.gpu_memory_utilization,
                0.75,
                0.70,
                0.65,
                0.60,
            ]
            seen = set()
            utils = [u for u in util_candidates if not (u in seen or seen.add(u)) and u <= 0.90]

            last_exc = None
            for u in utils:
                try:
                    log.info("Thu khoi tao vLLM voi gpu_memory_utilization=%.2f", u)
                    self._engine = LLM(
                        model=self.model_id,
                        dtype=self.dtype,
                        gpu_memory_utilization=u,
                        max_model_len=self.max_model_len,
                        seed=self.seed,
                        trust_remote_code=True,
                        enforce_eager=True,
                    )
                    log.info("Khoi tao vLLM thanh cong (gpu_memory_utilization=%.2f, enforce_eager=True)", u)
                    break
                except Exception as exc:
                    last_exc = exc
                    log.warning("vLLM khoi tao that bai voi gpu_memory_utilization=%.2f: %s", u, exc)
                    import gc
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

            if self._engine is None:
                log.warning("Khong the khoi tao vLLM (%s) -> Chuyen sang backend transformers", last_exc)
                self.backend = "transformers"
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
        if engine is None or self.backend != "vllm":
            return self._chat_transformers(messages, **overrides)
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
            kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            is_awq = "awq" in self.model_id.lower()
            is_gptq = "gptq" in self.model_id.lower()

            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
                if not is_awq and not is_gptq:
                    # Tu dong su dung 4-bit NF4 Quantization (bitsandbytes) cho model chua luong tu hoa (chi ton ~4.5GB VRAM cho 7B)
                    try:
                        from transformers import BitsAndBytesConfig  # type: ignore
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        log.info("Su dung 4-bit NF4 Quantization (BitsAndBytes) cho %s (tiet kiem VRAM ~4.5GB)", self.model_id)
                    except Exception as e:
                        log.warning("Khong nạp duoc bitsandbytes (%s) -> Dùng float16", e)
                        kwargs["torch_dtype"] = torch.float16
                else:
                    kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["device_map"] = "cpu"
                kwargs["torch_dtype"] = torch.float32

            self._engine = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

        tok = self.tokenizer
        text = self._apply_template(messages)
        # Cho phep do dai prompt toi da 3072 token (mo hinh 7B hoan toan du VRAM)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=3072).to(self._engine.device)

        temperature = overrides.get("temperature", self.temperature)
        # Cho phep toi da 512 token de sinh ma pandas phuc tap nhieu buoc
        max_tokens = min(int(overrides.get("max_tokens", self.max_tokens)), 512)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tok.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = self.top_p
        else:
            gen_kwargs["do_sample"] = False

        try:
            with torch.no_grad():
                out = self._engine.generate(**inputs, **gen_kwargs)
            gen = out[0][inputs["input_ids"].shape[1] :]
            decoded = tok.decode(gen, skip_special_tokens=True).strip()
            del inputs, out, gen
        except Exception as exc:
            log.warning("Loi generate transformers: %s -> Giai phong VRAM", exc)
            decoded = ""
            del inputs
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return decoded

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
