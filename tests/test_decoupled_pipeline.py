"""Kiem thu luong 2 Pha tach roi: RetrievalPipeline va run_from_retrieval."""

from pathlib import Path
import pandas as pd
import pytest

from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.pipeline.answer_pipeline import AnswerPipeline
from src.schemas import Question, RetrievalResult, RetrievedTable
from src.prompts.prompt_templates import format_linked_tables_note, format_unit_and_rounding_guidelines


def test_format_linked_tables_note():
    df1 = pd.DataFrame({"item": ["Doanh thu", "Gia von"], "value": [100, 60], "group_id": ["grp_1", "grp_1"]})
    df2 = pd.DataFrame({"item": ["Loi nhuan"], "value": [40], "group_id": ["grp_1"]})
    frames = {"df1": df1, "df2": df2}

    note = format_linked_tables_note(frames)
    assert "df1, df2" in note
    assert "pd.concat([df1, df2], ignore_index=True)" in note


def test_format_unit_and_rounding_guidelines():
    g1 = format_unit_and_rounding_guidelines("Lợi nhuận năm 2020 là bao nhiêu tỷ đồng?", asked_unit="ty_dong")
    assert "TỶ ĐỒNG" in g1
    assert "1e9" in g1

    g2 = format_unit_and_rounding_guidelines("Lãi tiền gửi là bao nhiêu triệu đồng?", asked_unit="trieu_dong")
    assert "TRIỆU ĐỒNG" in g2
    assert "1e6" in g2

    g3 = format_unit_and_rounding_guidelines("Vào năm nào doanh thu cao nhất?")
    assert "Hỏi NĂM" in g3
    assert "2019.0" in g3
