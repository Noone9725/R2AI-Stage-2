"""Test nhan dien bang tren 3 dialect OCR van ban (duong FALLBACK).

Kho THUC TE cua BTC ma hoa bang bang HTML inline, khong dung pipe/tab —
xem tests/test_html_table.py (fixture cat nguyen van tu data/raw) de biet
duong chinh. Cac fixture o day chi con bao ve nhanh fallback cho file
khong co the <table> nao (do trong mau 120 file: 1 file).

Cac fixture duoi day khong chua the <table> nao, nen `detect()` roi thang
xuong `_detect_text` — do la ly do goi `detect()` o day van kiem tra dung
nhanh van ban.
"""

from __future__ import annotations

import pytest

from src.extraction.table_detector import Dialect, TableDetector
from src.schemas import RawDocument


def _doc(text: str, doc_id: str = "AAA_financial_statements_2020_consolidated") -> RawDocument:
    return RawDocument(doc_id=doc_id, path=f"raw/{doc_id}.txt", text=text)


PIPE_TXT = """CÔNG TY CỔ PHẦN AAA
BẢNG CÂN ĐỐI KẾ TOÁN
Đơn vị: triệu đồng

| Chỉ tiêu | 2020 | 2019 |
| --- | --- | --- |
| Tài sản ngắn hạn | 1.234.567 | 1.100.000 |
| Tiền và tương đương tiền | 234.567 | 200.000 |
| TỔNG CỘNG TÀI SẢN | 2.000.000 | 1.800.000 |
"""

TAB_TXT = "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH\n\nChỉ tiêu\t2020\t2019\nDoanh thu thuần\t5.000.000\t4.500.000\nLợi nhuận gộp\t1.000.000\t900.000\n"

WS_TXT = """BÁO CÁO LƯU CHUYỂN TIỀN TỆ

Chỉ tiêu                          2020          2019
Lưu chuyển từ hoạt động kinh doanh    500.000       450.000
Lưu chuyển từ hoạt động đầu tư       (200.000)     (150.000)
"""


class TestDialectDetection:
    @pytest.mark.parametrize(
        "text,dialect",
        [
            (PIPE_TXT, Dialect.PIPE),
            (TAB_TXT, Dialect.TAB),
            (WS_TXT, Dialect.WHITESPACE),
        ],
    )
    def test_pick_dialect(self, text: str, dialect: Dialect) -> None:
        det = TableDetector(min_rows=2, min_cols=2)
        assert det._pick_dialect(text.splitlines()) is dialect

    def test_empty_defaults_to_whitespace(self) -> None:
        det = TableDetector(min_rows=2, min_cols=2)
        assert det._pick_dialect([]) is Dialect.WHITESPACE


class TestDetect:
    @pytest.mark.parametrize("text", [PIPE_TXT, TAB_TXT, WS_TXT])
    def test_finds_at_least_one_table(self, text: str) -> None:
        tables = TableDetector(min_rows=2, min_cols=2).detect(_doc(text))
        assert tables, "khong cat duoc bang nao"
        assert all(len(t.rows) >= 2 for t in tables)

    def test_no_table_in_prose(self) -> None:
        prose = "Công ty được thành lập năm 2005 theo giấy phép số 123.\nNgành nghề chính là sản xuất bao bì.\n"
        assert TableDetector(min_rows=2, min_cols=2).detect(_doc(prose)) == []

    def test_position_starts_at_one(self) -> None:
        tables = TableDetector(min_rows=2, min_cols=2).detect(_doc(PIPE_TXT))
        assert tables[0].position == 1

    def test_table_ref_format(self) -> None:
        """`relevant_tables` phai la '<doc_id>|<position>' — mat la mat 1/3 diem."""
        doc = _doc(PIPE_TXT)
        table = TableDetector(min_rows=2, min_cols=2).detect(doc)[0]
        assert table.table_ref == f"{doc.doc_id}|{table.position}"
        assert table.doc_id == doc.doc_id

    def test_cells_are_split(self) -> None:
        table = TableDetector(min_rows=2, min_cols=2).detect(_doc(PIPE_TXT))[0]
        flat = [c for row in table.rows for c in row]
        assert "1.234.567" in flat
        assert any("Tài sản ngắn hạn" in c for c in flat)

    def test_min_rows_filters_small_blocks(self) -> None:
        tiny = "Chỉ tiêu | 2020\nTài sản | 1.000\n"
        assert TableDetector(min_rows=5, min_cols=2).detect(_doc(tiny)) == []

    def test_separator_row_dropped(self) -> None:
        """Dong '| --- | --- |' khong duoc thanh mot hang du lieu."""
        table = TableDetector(min_rows=2, min_cols=2).detect(_doc(PIPE_TXT))[0]
        for row in table.rows:
            assert not all(set(c) <= set("-:= ") for c in row if c), row


class TestSection:
    @pytest.mark.parametrize(
        "context,section",
        [
            ("BẢNG CÂN ĐỐI KẾ TOÁN", "balance_sheet"),
            ("Balance Sheet", "balance_sheet"),
            ("BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH", "income_statement"),
            ("BÁO CÁO LƯU CHUYỂN TIỀN TỆ", "cash_flow"),
            ("THUYẾT MINH BÁO CÁO TÀI CHÍNH", "notes"),
            ("Danh sách cổ đông", None),
        ],
    )
    def test_detect_section(self, context: str, section: str | None) -> None:
        assert TableDetector._detect_section(context) == section

    def test_section_on_real_block(self) -> None:
        table = TableDetector(min_rows=2, min_cols=2).detect(_doc(PIPE_TXT))[0]
        assert table.section == "balance_sheet"


class TestSplitRow:
    def test_pipe_strips_edges(self) -> None:
        row = TableDetector._split_row("| a | b | c |", Dialect.PIPE)
        assert row == ["a", "b", "c"]

    def test_tab(self) -> None:
        assert TableDetector._split_row("a\tb\tc", Dialect.TAB) == ["a", "b", "c"]

    def test_whitespace_needs_two_spaces(self) -> None:
        """Mot dau cach la trong cung mot o (ten chi tieu co khoang trang)."""
        row = TableDetector._split_row("Tài sản ngắn hạn    1.000    900", Dialect.WHITESPACE)
        assert row == ["Tài sản ngắn hạn", "1.000", "900"]


class TestContext:
    def test_title_from_context(self) -> None:
        table = TableDetector(min_rows=2, min_cols=2).detect(_doc(PIPE_TXT))[0]
        assert table.title, "khong tim duoc tieu de"

    def test_context_before_captured(self) -> None:
        """context_before phai giu dong don vi — can cho detect_unit."""
        table = TableDetector(min_rows=2, min_cols=2).detect(_doc(PIPE_TXT))[0]
        assert "triệu đồng" in table.context_before.lower()
