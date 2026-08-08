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
        schemas = pt.format_schemas(frames)
        last: ExecutionResult | None = None

        for attempt in range(1, self.max_attempts + 1):
            result = self.sandbox.run(code, frames, attempt=attempt)
            if result.success:
                if attempt > 1:
                    log.info("Sua thanh cong sau %d luot", attempt)
                query.pandas_query = code
                query.attempt = attempt
                return result, query

            last = result
            log.debug("Luot %d that bai: %s", attempt, result.error)

            if attempt == self.max_attempts:
                break

            prompt = pt.render(
                "self_repair",
                question=question_text,
                variables=variables,
                schemas=schemas,
                code=code,
                error=result.error,
                hints=diagnose(result, frames),
            )
            try:
                response = self.llm.generate(prompt, system=pt.SYSTEM_PANDAS)
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM loi khi sua code: %s", exc)
                break

            new_code = extract_code(response)
            if not new_code or new_code == code:
                log.debug("LLM khong dua ra code moi — dung sua")
                break
            code = new_code

        assert last is not None
        query.pandas_query = code
        query.attempt = last.attempt
        return last, query
