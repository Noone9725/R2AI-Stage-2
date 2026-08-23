"""Chay code pandas do LLM sinh, trong pham vi kiem soat.

Khong phai sandbox bao mat that (khong the, code chay cung process), ma la
sandbox VE TAI NGUYEN VA TEN: chan import, chan builtin nguy hiem, gioi han
thoi gian, va chi cap dung nhung DataFrame duoc phep. Muc tieu la mot query
sai khong lam sap ca run 100+ cau hoi.
"""

from __future__ import annotations

import ast
import math
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_settings
from ..schemas import ExecutionResult
from ..utils.logging import get_logger

log = get_logger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

# AST node bi cam — chan truoc khi chay, khong doi loi runtime
_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)
_FORBIDDEN_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input",
    "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "breakpoint", "memoryview",
})
_FORBIDDEN_ATTRS = frozenset({
    "system", "popen", "remove", "unlink", "rmtree", "to_csv", "to_pickle",
    "__class__", "__bases__", "__subclasses__", "__globals__", "__builtins__",
})

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "print": print, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "ZeroDivisionError": ZeroDivisionError, "KeyError": KeyError,
    "IndexError": IndexError, "ValueError": ValueError, "TypeError": TypeError,
    "Exception": Exception, "NameError": NameError,
}


class UnsafeCodeError(Exception):
    pass


def extract_code(response: str) -> str:
    """Lay code tu response LLM. Uu tien block ```python...``` va kiem tra tinh hop le qua AST."""
    if not response or not response.strip():
        return ""

    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
    candidates = blocks if blocks else [response]

    for cand in candidates:
        cand = cand.strip()
        try:
            ast.parse(cand)
            return cand
        except SyntaxError:
            pass

        # Loc cac dong code hop le
        cleaned = _clean_code_lines(cand)
        if cleaned:
            try:
                ast.parse(cleaned)
                return cleaned
            except SyntaxError:
                pass

    return ""


def _clean_code_lines(text: str) -> str:
    """Loai bo cac dong van ban tu nhien, giai thich, hoac bang bieu markdown."""
    lines = text.splitlines()
    valid_lines = []
    code_started = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if code_started:
                valid_lines.append(line)
            continue
        if stripped.startswith("```") or stripped.startswith("==="):
            continue
        if stripped.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
            continue
        if any(stripped.startswith(prefix) for prefix in [
            "Đề xuất", "Để sửa", "Các giá trị", "Mẫu dữ liệu", "Lược đồ",
            "Cột:", "dòng:", "ticker ", "Lưu ý:", "Giải thích:", "Kết quả:"
        ]):
            continue

        # Nhan dien dong code Python hop le
        if re.match(r"^\s*(?:result\s*=|df\d*\s*=|all_df\s*=|dfs\s*=|sub\d*\s*=|pat_\w*\s*=|equity_\w*\s*=|import |from |for |if |else:|elif |def |return |with |try:|except|#|[a-z_]\w*\s*=)", line):
            code_started = True
            valid_lines.append(line)
        elif code_started and not any(c in line for c in ["Đề xuất", "Để sửa", "những cột không cần"]):
            valid_lines.append(line)

    return "\n".join(valid_lines).strip()


_ALLOWED_MODULES = frozenset({"pandas", "numpy", "math"})


class PandasSandbox:
    def __init__(self, timeout_sec: float | None = None):
        cfg = get_settings().execution
        self.timeout_sec = float(
            timeout_sec if timeout_sec is not None else cfg.get("timeout_sec", 15)
        )
        self.allow_import = bool(cfg.get("allow_import", True) or cfg.get("allow_imports", True))

    # ── static check ──────────────────────────────────────

    def check(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise UnsafeCodeError(f"SyntaxError: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod not in _ALLOWED_MODULES:
                        raise UnsafeCodeError(f"Khong duoc import module: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod not in _ALLOWED_MODULES:
                    raise UnsafeCodeError(f"Khong duoc import tu module: {node.module}")
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                raise UnsafeCodeError(f"Ten bi cam: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
                raise UnsafeCodeError(f"Thuoc tinh bi cam: .{node.attr}")

        if "result" not in {
            t.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }:
            raise UnsafeCodeError("Code khong gan bien `result`")

    # ── run ───────────────────────────────────────────────

    def run(
        self,
        code: str,
        frames: dict[str, pd.DataFrame],
        attempt: int = 1,
    ) -> ExecutionResult:
        started = time.perf_counter()

        try:
            self.check(code)
        except UnsafeCodeError as exc:
            return ExecutionResult(
                success=False, value=None, raw_value=None,
                error=f"UnsafeCodeError: {exc}", error_type="UnsafeCodeError",
                code=code,
                elapsed_ms=(time.perf_counter() - started) * 1000, attempt=attempt,
            )

        # Smart wrapper cho pd.read_csv neu LLM van co tinh goi read_csv
        def _safe_read_csv(filepath: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
            p_str = str(filepath).replace("\\", "/")
            fname = Path(p_str).name
            for v, df in frames.items():
                if v == p_str or fname in p_str:
                    return df.copy()
            # Thu doc tu data/processed neu file ton tai
            cand = Path("data/processed") / fname
            if cand.exists():
                return pd.read_csv(cand, *args, **kwargs)
            # Tra ve df dau tien neu co
            if frames:
                return next(iter(frames.values())).copy()
            return pd.DataFrame()

        # Tao bien df tong hop toan bo cac bang
        combined_df = pd.concat([d for d in frames.values() if isinstance(d, pd.DataFrame)], ignore_index=True) if frames else pd.DataFrame()

        # Tạo module pd an toàn
        safe_pd = pd
        # Injects frames và helper
        env: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": safe_pd, "numpy": np, "np": np, "math": math,
            "df": combined_df.copy(),
            "all_df": combined_df.copy(),
            **{name: df.copy() for name, df in frames.items()},
        }

        with _time_limit(self.timeout_sec):
            try:
                exec(compile(code, "<generated>", "exec"), env)   # noqa: S102
            except Exception as exc:  # noqa: BLE001
                return ExecutionResult(
                    success=False, value=None, raw_value=None,
                    error=f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__, code=code,
                    elapsed_ms=(time.perf_counter() - started) * 1000, attempt=attempt,
                )

        raw = env.get("result")
        elapsed = (time.perf_counter() - started) * 1000
        value, err = coerce_number(raw)

        if value is None:
            return ExecutionResult(
                success=False, value=None, raw_value=raw, error=err,
                error_type="ResultTypeError", code=code,
                elapsed_ms=elapsed, attempt=attempt,
            )

        # Kiem tra neu result = 0.0 do filter sub bi rong (empty subset)
        if value == 0.0:
            empty_subs = [
                k for k, v in env.items()
                if isinstance(v, pd.DataFrame) and k not in frames and k not in ("df", "all_df") and v.empty
            ]
            if empty_subs:
                return ExecutionResult(
                    success=False, value=0.0, raw_value=raw,
                    error=f"EmptyFilterError: Tap du lieu con `{empty_subs[0]}` bi rong (0 dong) do dieu kien loc qua chat. Hay bo bot dieu kien (vd: bo loc 'period' hoac rut ngan tu khoa item).",
                    error_type="EmptyFilterError", code=code,
                    elapsed_ms=elapsed, attempt=attempt,
                )

        return ExecutionResult(
            success=True, value=value, raw_value=raw, error="", error_type="",
            code=code, elapsed_ms=elapsed, attempt=attempt,
        )


def coerce_number(raw: Any) -> tuple[float | None, str]:
    """Ep ket qua ve float. Chap nhan Series/array 1 phan tu."""
    if raw is None:
        return None, "Bien `result` khong duoc gan gia tri"

    if isinstance(raw, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        if len(raw) == 1:
            return coerce_number(list(raw)[0])
        return None, f"`result` co {len(raw)} phan tu, can dung 1 so"

    if isinstance(raw, pd.DataFrame):
        if raw.shape == (1, 1):
            return coerce_number(raw.iloc[0, 0])
        return None, f"`result` la DataFrame {raw.shape}, can dung 1 so"

    if isinstance(raw, bool):
        return float(raw), ""

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"`result` kieu {type(raw).__name__} khong doi duoc sang so"

    if math.isnan(value):
        return None, "`result` la NaN"
    if math.isinf(value):
        return None, "`result` la vo cuc (co the chia cho 0)"
    return value, ""


# ── timeout ───────────────────────────────────────────────


class _time_limit:
    """Gioi han thoi gian. Dung SIGALRM tren POSIX; Windows khong co
    SIGALRM nen chi la no-op — pandas query hiem khi treo, va tach process
    cho moi query se dat hon nhieu so voi rui ro.
    """

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._old: Any = None

    def __enter__(self) -> "_time_limit":
        try:
            import signal

            def _raise(_sig: int, _frm: Any) -> None:
                raise TimeoutError(f"Vuot {self.seconds}s")

            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        except (ImportError, AttributeError, ValueError):
            self._old = None
        return self

    def __exit__(self, *_exc: Any) -> bool:
        if self._old is not None:
            import signal

            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
        return False
