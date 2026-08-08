"""Test schema bai nop — hang rao cuoi truoc khi tieu mot luot nop.

Public test 10 luot/ngay, private test 5 luot TONG. Mot bai bi tu choi vi
sai schema la mat mot luot vo ich, nen validator phai bat het loi dinh dang
truoc khi zip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas import Evidence, SubmissionItem
from src.submission import validate_submission


def _row(**over) -> dict:
    row = {
        "id": 1,
        "question": "Tổng tài sản của AAA năm 2020 là bao nhiêu?",
        "answer": 1234567.0,
        "relevant_docs": ["AAA_financial_statements_2020_consolidated"],
        "relevant_tables": ["AAA_financial_statements_2020_consolidated|3"],
        "evidence": [{"variable": "df_aaa", "csv_path": "data/aaa_2020_t3.csv"}],
        "pandas_query": "result = df_aaa['value'].sum()",
    }
    row.update(over)
    return row


class TestValidRow:
    def test_baseline_is_clean(self) -> None:
        rep = validate_submission([_row()])
        assert rep.ok, rep.render()
        assert not rep.warnings, rep.render()

    def test_multiple_items(self) -> None:
        rep = validate_submission([_row(id=1), _row(id=2), _row(id=3)])
        assert rep.ok, rep.render()


class TestStructure:
    def test_not_a_list(self) -> None:
        assert not validate_submission({"id": 1}).ok        # type: ignore[arg-type]

    def test_empty_list(self) -> None:
        assert not validate_submission([]).ok

    @pytest.mark.parametrize(
        "key",
        ["id", "question", "answer", "relevant_docs",
         "relevant_tables", "evidence", "pandas_query"],
    )
    def test_missing_key(self, key: str) -> None:
        row = _row()
        row.pop(key)
        rep = validate_submission([row])
        assert not rep.ok
        assert any(key in e for e in rep.errors), rep.render()


class TestTypes:
    @pytest.mark.parametrize("bad", ["1", 1.5, None, True])
    def test_id_must_be_int(self, bad: object) -> None:
        assert not validate_submission([_row(id=bad)]).ok

    def test_duplicate_id(self) -> None:
        rep = validate_submission([_row(id=7), _row(id=7)])
        assert not rep.ok
        assert any("trung" in e for e in rep.errors)

    @pytest.mark.parametrize("bad", ["1234", None, True, [1]])
    def test_answer_must_be_number(self, bad: object) -> None:
        assert not validate_submission([_row(answer=bad)]).ok

    def test_answer_int_is_ok(self) -> None:
        assert validate_submission([_row(answer=1234)]).ok

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_rejected(self, bad: float) -> None:
        """json.dumps cho ra NaN/Infinity — khong phai JSON hop le."""
        assert not validate_submission([_row(answer=bad)]).ok

    def test_empty_question(self) -> None:
        assert not validate_submission([_row(question="")]).ok

    def test_pandas_query_must_be_str(self) -> None:
        assert not validate_submission([_row(pandas_query=None)]).ok


class TestRelevantDocs:
    def test_txt_suffix_rejected(self) -> None:
        """doc_id la ten file BO duoi .txt."""
        rep = validate_submission([_row(relevant_docs=["AAA_2020.txt"],
                                        relevant_tables=["AAA_2020.txt|1"])])
        assert not rep.ok
        assert any(".txt" in e for e in rep.errors)

    def test_not_list_of_str(self) -> None:
        assert not validate_submission([_row(relevant_docs="AAA_2020")]).ok

    def test_empty_docs_warns_only(self) -> None:
        rep = validate_submission([_row(relevant_docs=[], relevant_tables=[])])
        assert rep.ok
        assert rep.warnings


class TestRelevantTables:
    def test_missing_pipe(self) -> None:
        rep = validate_submission([_row(relevant_tables=["AAA_financial_statements_2020_consolidated"])])
        assert not rep.ok

    def test_position_not_numeric(self) -> None:
        rep = validate_submission(
            [_row(relevant_tables=["AAA_financial_statements_2020_consolidated|abc"])]
        )
        assert not rep.ok

    def test_doc_not_in_relevant_docs_warns(self) -> None:
        rep = validate_submission([_row(relevant_tables=["BBB_2020_consolidated|1"])])
        assert rep.ok
        assert any("khong nam trong relevant_docs" in w for w in rep.warnings)


class TestEvidenceCsvPath:
    def test_backslash_rejected(self) -> None:
        rep = validate_submission(
            [_row(evidence=[{"variable": "df_aaa", "csv_path": "data\\aaa.csv"}])]
        )
        assert not rep.ok

    @pytest.mark.parametrize("bad", ["/data/aaa.csv", "C:/data/aaa.csv"])
    def test_absolute_rejected(self, bad: str) -> None:
        rep = validate_submission([_row(evidence=[{"variable": "df_aaa", "csv_path": bad}])])
        assert not rep.ok

    def test_prefix_required(self) -> None:
        """Moi csv_path phai bat dau bang 'data/'."""
        rep = validate_submission(
            [_row(evidence=[{"variable": "df_aaa", "csv_path": "csv/aaa.csv"}])]
        )
        assert not rep.ok

    def test_missing_field(self) -> None:
        assert not validate_submission([_row(evidence=[{"variable": "df_aaa"}])]).ok

    def test_variable_absent_from_code_warns(self) -> None:
        rep = validate_submission([_row(pandas_query="result = df_other['value'].sum()")])
        assert rep.ok
        assert any("df_aaa" in w for w in rep.warnings)


class TestSoftSignals:
    def test_empty_code_warns(self) -> None:
        rep = validate_submission([_row(pandas_query="", evidence=[])])
        assert rep.ok
        assert any("Execution" in w for w in rep.warnings)

    def test_code_without_result_warns(self) -> None:
        rep = validate_submission([_row(pandas_query="df_aaa['value'].sum()")])
        assert rep.ok
        assert any("result" in w for w in rep.warnings)


class TestExpectedIds:
    def test_missing_question_is_error(self) -> None:
        rep = validate_submission([_row(id=1)], expected_ids=[1, 2, 3])
        assert not rep.ok
        assert any("Thieu" in e for e in rep.errors)

    def test_exact_match_ok(self) -> None:
        rep = validate_submission([_row(id=1), _row(id=2)], expected_ids=[1, 2])
        assert rep.ok, rep.render()


class TestSubmissionItem:
    def test_to_dict_passes_validation(self) -> None:
        """Dataclass -> dict phai qua duoc chinh validator cua minh."""
        item = SubmissionItem(
            id=1,
            question="Tổng tài sản của AAA năm 2020?",
            answer=1234567.0,
            relevant_docs=["AAA_financial_statements_2020_consolidated"],
            relevant_tables=["AAA_financial_statements_2020_consolidated|3"],
            evidence=[Evidence(variable="df_aaa", csv_path="data/aaa_2020_t3.csv")],
            pandas_query="result = df_aaa['value'].sum()",
        )
        rep = validate_submission([item.to_dict()])
        assert rep.ok, rep.render()

    def test_to_dict_is_json_serializable(self) -> None:
        item = SubmissionItem(id=1, question="q", answer=0.0)
        text = json.dumps([item.to_dict()], ensure_ascii=False)
        assert json.loads(text)[0]["id"] == 1

    def test_default_shape(self) -> None:
        """Cau tra loi that bai van phai giu du 7 key."""
        row = SubmissionItem(id=1, question="q", answer=0.0).to_dict()
        for key in ("id", "question", "answer", "relevant_docs",
                    "relevant_tables", "evidence", "pandas_query"):
            assert key in row


class TestRootDirCheck:
    def test_missing_csv_file_flagged(self, tmp_path: Path) -> None:
        rep = validate_submission([_row()], root_dir=tmp_path)
        assert rep.errors or rep.warnings, "khong bao gi khi thieu file CSV"

    def test_existing_csv_ok(self, tmp_path: Path) -> None:
        csv = tmp_path / "data" / "aaa_2020_t3.csv"
        csv.parent.mkdir(parents=True)
        csv.write_text("ticker,year,value\nAAA,2020,1\n", encoding="utf-8")
        rep = validate_submission([_row()], root_dir=tmp_path)
        assert rep.ok, rep.render()
