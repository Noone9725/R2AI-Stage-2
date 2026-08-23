"""Vong sinh code -> chay -> loi -> sua -> chay lai.

Diem Execution Accuracy phu thuoc truc tiep vao vong nay: mot loi
KeyError don gian van tinh la sai hoan toan neu khong sua. Thuc nghiem
tren cac bo Text-to-SQL/Pandas: 2-3 luot sua vot lai 15-25% cau hoi.
"""

from __future__ import annotations

import pandas as pd

from ..config import get_settings
from ..llm.llm_client import LLMClient
from ..prompts import prompt_templates as pt
from ..schemas import ExecutionResult, GeneratedQuery
from ..utils.logging import get_logger
from .sandbox import PandasSandbox, extract_code
from .validator import diagnose

log = get_logger(__name__)


class SelfRepairExecutor:
    def __init__(
        self,
        llm: LLMClient | None = None,
        sandbox: PandasSandbox | None = None,
        max_attempts: int | None = None,
    ):
        cfg = get_settings().execution
        self.llm = llm or LLMClient()
        self.sandbox = sandbox or PandasSandbox()
        self.max_attempts = int(
            max_attempts if max_attempts is not None
            else cfg.get("max_repair_attempts", 3)
        )

    def run(
        self,
        question_text: str,
        query: GeneratedQuery,
        frames: dict[str, pd.DataFrame],
    ) -> tuple[ExecutionResult, GeneratedQuery]:
        code = query.pandas_query
        variables = pt.format_variables(query.var_to_csv)
        schemas = pt.format_schemas(frames, question_text=question_text)
        last: ExecutionResult | None = None

        if not code or not code.strip():
            log.info("LLM chua sinh duoc code hop le o buoc dau -> Khoi dong Self-Repair tao ma...")
            # Tao prompt ban dau de yeu cau viet code
            hints = "Chưa có code hợp lệ. Hãy viết code pandas gán kết quả vào biến `result`."
            prompt = pt.render(
                "self_repair",
                question=question_text,
                variables=variables,
                schemas=schemas,
                code="# Chưa có code",
                error="No valid code generated initially",
                hints=hints,
            )
            try:
                response = self.llm.generate(prompt, system=pt.SYSTEM_PANDAS)
                code = extract_code(response)
                if not code:
                    from .sandbox import _clean_code_lines
                    code = _clean_code_lines(response)
            except Exception as exc:  # noqa: BLE001
                log.info("LLM loi khi sinh code khoi tao: %s", exc)
                code = ""

        if not code:
            log.info("Khong the khoi tao ma code hop le tu LLM -> Dung fallback 0.0")
            res = ExecutionResult(success=False, value=0.0, raw_value=0.0, error="No valid code generated", attempt=0)
            query.pandas_query = ""
            return res, query

        for attempt in range(1, self.max_attempts + 1):
            log.info("--- [Luot %d/%d] Chay ma Pandas: ---", attempt, self.max_attempts)
            for line in code.strip().splitlines():
                log.info("  | %s", line)

            result = self.sandbox.run(code, frames, attempt=attempt)
            if result.success:
                if attempt == 1:
                    log.info("==> [Thanh cong L1] -> result=%s", result.value)
                else:
                    log.info("==> [Sua Thanh cong L%d] -> result=%s", attempt, result.value)
                query.pandas_query = code
                query.attempt = attempt
                return result, query

            last = result
            hint_text = diagnose(result, frames)
            log.info("==> [L%d That bai] Loi: %s (%s)", attempt, result.error, result.error_type)
            log.info("    Chan doan: %s", hint_text.replace("\n", " | "))

            if attempt == self.max_attempts:
                break

            prompt = pt.render(
                "self_repair",
                question=question_text,
                variables=variables,
                schemas=schemas,
                code=code,
                error=result.error,
                hints=hint_text,
            )
            try:
                response = self.llm.generate(prompt, system=pt.SYSTEM_PANDAS)
            except Exception as exc:  # noqa: BLE001
                log.info("LLM loi khi sua code o luot %d: %s", attempt, exc)
                break

            new_code = extract_code(response)
            if not new_code:
                from .sandbox import _clean_code_lines
                new_code = _clean_code_lines(response)
            if not new_code:
                log.info("LLM khong sinh duoc ma code moi o luot %d", attempt)
                continue
            code = new_code

        assert last is not None
        log.info("==> Ket thuc %d luot: THAT BAI (%s) -> Bao toan ma sinh duoc va gan fallback 0.0", self.max_attempts, last.error)
        query.pandas_query = code
        query.attempt = last.attempt
        return last, query
