"""Sinh code pandas tu cau hoi + cac bang da chon.

Nhiem vu chinh o day khong phai goi LLM, ma la DON PROMPT: nap DataFrame,
gan ten bien on dinh (df1, df2...), liet ke san danh sach `item` de LLM
copy thay vi doan ten chi tieu tieng Viet.
"""

from __future__ import annotations

import pandas as pd

from ..llm.llm_client import LLMClient
from ..normalization.csv_writer import CsvWriter
from ..prompts import prompt_templates as pt
from ..schemas import GeneratedQuery, Question, RetrievalResult
from ..utils.logging import get_logger
from ..execution.sandbox import extract_code

log = get_logger(__name__)


class PandasGenerator:
    def __init__(self, llm: LLMClient | None = None, csv_writer: CsvWriter | None = None):
        self.llm = llm or LLMClient()
        self.csv = csv_writer or CsvWriter()

    # ── load frames ───────────────────────────────────────

    def load_frames(self, retrieval: RetrievalResult) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Tra ve ({var: df}, {var: csv_path}).

        Bang khong doc duoc thi bo qua chu khong ra loi — mot CSV hong khong
        nen keo ca cau hoi ve 0 khi con bang khac dung duoc.
        """
        frames: dict[str, pd.DataFrame] = {}
        var_to_csv: dict[str, str] = {}

        for i, table in enumerate(retrieval.tables, start=1):
            if not table.csv_path:
                continue
            local = self.csv.resolve_local(table.csv_path)
            try:
                df = pd.read_csv(local, encoding="utf-8-sig")
            except Exception as exc:  # noqa: BLE001
                log.warning("Khong doc duoc %s: %s", local, exc)
                continue
            var = f"df{i}"
            frames[var] = df
            var_to_csv[var] = table.csv_path

        return frames, var_to_csv

    # ── generate ──────────────────────────────────────────

    def generate(
        self,
        question: Question,
        retrieval: RetrievalResult,
        frames: dict[str, pd.DataFrame] | None = None,
        var_to_csv: dict[str, str] | None = None,
    ) -> GeneratedQuery:
        if frames is None or var_to_csv is None:
            frames, var_to_csv = self.load_frames(retrieval)

        if not frames:
            return GeneratedQuery(
                pandas_query="", var_to_csv={},
                reasoning="khong nap duoc bang nao", attempt=0, raw_response="",
            )

        prompt = pt.render(
            "pandas_gen",
            question=question.question,
            variables=pt.format_variables(var_to_csv),
            schemas=pt.format_schemas(frames),
            formulas=pt.format_formulas(question.metrics),
            period_hint=pt.format_period_hint(question),
        )

        try:
            response = self.llm.generate(prompt, system=pt.SYSTEM_PANDAS)
        except Exception as exc:  # noqa: BLE001
            log.error("LLM loi khi sinh code Q%d: %s", question.id, exc)
            return GeneratedQuery(
                pandas_query="", var_to_csv=var_to_csv,
                reasoning=f"loi LLM: {exc}", attempt=0, raw_response="",
            )

        return GeneratedQuery(
            pandas_query=extract_code(response),
            var_to_csv=var_to_csv,
            reasoning="",
            attempt=1,
            raw_response=response,
        )
