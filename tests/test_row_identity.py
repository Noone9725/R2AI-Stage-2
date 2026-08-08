"""Test row identity: row_idx + column_label + item_code — Phase 2.6.

Phase 2.5 xac dinh root cause = ROW_IDENTITY, chia hai:
  68.2% item lap nhieu dong hop le (dong con) -> can item_code
  31.8% bang co truc cot la CATEGORY          -> can column_label
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.embeddings.table_card import TableCardBuilder
from src.normalization.schema_std import STANDARD_COLUMNS, SchemaStandardizer
from src.prompts import prompt_templates as pt
from src.schemas import ExtractedTable


def _table(rows: list[list[str]], columns: list[str], **kw) -> ExtractedTable:
    t = ExtractedTable(doc_id=kw.pop("doc_id", "DOC_2017"), position=kw.pop("position", 2), **kw)
    t.df = pd.DataFrame(rows, columns=columns)
    return t


def _std(table: ExtractedTable, year: int | None = 2017) -> pd.DataFrame:
    return SchemaStandardizer().standardize(table, ticker="AAA", fallback_year=year)


# ── A. Repeated item, different item_code ─────────────────


def test_A_item_lap_khac_item_code() -> None:
    """Bang can doi that: '- Nguyen gia' la dong con cua TSCD huu hinh (222)
    va TSCD vo hinh (228)."""
    t = _table(
        [
            ["1. Tài sản cố định hữu hình", "221", "1.843.161.063.205", "1.232.275.819.024"],
            ["- Nguyên giá", "222", "2.301.366.557.311", "1.573.025.768.925"],
            ["2. Tài sản cố định vô hình", "227", "73.473.172.562", "72.638.218.007"],
            ["- Nguyên giá", "228", "78.363.572.029", "75.966.426.029"],
        ],
        ["TÀI SẢN", "Mã số", "31/12/2017", "01/01/2017"],
    )
    long = _std(t)
    ng = long[long["item"] == "- Nguyên giá"]

    assert set(ng["item_code"]) == {"222", "228"}
    # them item_code vao key -> het ambiguous
    amb = ng.groupby(["item_code", "item", "year", "period"])["value"].nunique()
    assert (amb > 1).sum() == 0

    closing_222 = ng[(ng["item_code"] == "222") & (ng["period"] == "closing")]
    assert closing_222["value"].iloc[0] == pytest.approx(2_301_366_557_311)


# ── B. Category-axis table ────────────────────────────────


def test_B_bang_truc_category_giu_du_cot() -> None:
    """Truoc fix: ca bang chi lay MOT o so dau tien, mat 6 cot con lai."""
    t = _table(
        [
            ["Nguyên giá", "100", "200", "300"],
            ["Hao mòn luỹ kế", "10", "20", "30"],
        ],
        ["Hạng mục", "Nhà cửa", "Máy móc", "Phương tiện"],
    )
    long = _std(t)

    ng = long[long["item"] == "Nguyên giá"]
    assert len(ng) == 3, "phai giu du 3 cot category"
    assert set(ng["column_label"]) == {"Nhà cửa", "Máy móc", "Phương tiện"}
    assert set(ng["value"]) == {100.0, 200.0, 300.0}

    # column_label lam key -> khong con ambiguous
    amb = ng.groupby(["item", "year", "period", "column_label"])["value"].nunique()
    assert (amb > 1).sum() == 0


def test_B2_cot_thuyet_minh_khong_thanh_category() -> None:
    """Cot 'Thuyet minh' chua so hieu note, khong phai du lieu tai chinh."""
    t = _table(
        [["Tiền", "110", "5.1", "100.000", "90.000"]],
        ["Chỉ tiêu", "Mã số", "Thuyết minh", "31/12/2017", "01/01/2017"],
    )
    long = _std(t)
    assert "Thuyết minh" not in set(long["column_label"])
    assert set(long["column_label"]) == {"31/12/2017", "01/01/2017"}


# ── C. Time-axis table ────────────────────────────────────


def test_C_truc_thoi_gian_van_traceable() -> None:
    t = _table(
        [["Nguyên giá", "1.000", "900"]],
        ["Chỉ tiêu", "31/12/2023", "01/01/2023"],
    )
    long = _std(t, year=2023)

    closing = long[long["period"] == "closing"].iloc[0]
    opening = long[long["period"] == "opening"].iloc[0]
    assert (closing["year"], closing["column_label"]) == (2023, "31/12/2023")
    assert (opening["year"], opening["column_label"]) == (2022, "01/01/2023")
    assert closing["source_date"] == "31/12/2023"


# ── D. Repeated generic label ─────────────────────────────


def test_D_tong_cong_lap_khong_bi_gop() -> None:
    t = _table(
        [
            ["Khoản mục A", "100"],
            ["TỔNG CỘNG", "100"],
            ["Khoản mục B", "250"],
            ["TỔNG CỘNG", "250"],
        ],
        ["Chỉ tiêu", "31/12/2017"],
    )
    long = _std(t)
    tc = long[long["item"] == "TỔNG CỘNG"]

    assert len(tc) == 2, "khong duoc collapse"
    assert sorted(tc["row_idx"]) == [1, 3], "row_idx phai truy nguoc dung dong goc"
    assert sorted(tc["value"]) == [100.0, 250.0]


def test_D2_row_idx_khop_vi_tri_dong_goc() -> None:
    t = _table(
        [["A", "1"], ["B", "2"], ["C", "3"]],
        ["Chỉ tiêu", "2017"],
    )
    long = _std(t)
    assert dict(zip(long["item"], long["row_idx"])) == {"A": 0, "B": 1, "C": 2}


def test_D3_row_idx_cong_column_label_la_deterministic() -> None:
    """Invariant: (row_idx, column_label) xac dinh duy nhat mot o nguon."""
    t = _table(
        [["Nguyên giá", "100", "200"], ["Nguyên giá", "300", "400"]],
        ["Hạng mục", "Nhà cửa", "Máy móc"],
    )
    long = _std(t)
    key = long.groupby(["row_idx", "column_label"]).size()
    assert (key > 1).sum() == 0
    assert len(long) == 4


# ── E. Table ID khong doi ─────────────────────────────────


def test_E_table_ref_giu_nguyen() -> None:
    t = _table([["A", "1"]], ["Chỉ tiêu", "2017"], doc_id="AAA_2017_consolidated", position=44)
    long = _std(t)
    assert t.table_ref == "AAA_2017_consolidated|44"
    assert set(long["table_ref"]) == {"AAA_2017_consolidated|44"}


def test_E2_them_cot_khong_sinh_position_moi() -> None:
    """Bang category-axis sinh nhieu dong hon nhung VAN mot table_ref."""
    t = _table(
        [["Nguyên giá", "1", "2", "3", "4"]],
        ["Hạng mục", "A", "B", "C", "D"],
        doc_id="DOC", position=7,
    )
    long = _std(t)
    assert len(long) == 4
    assert long["table_ref"].nunique() == 1
    assert long["table_ref"].iloc[0] == "DOC|7"


# ── F. Khong regression ───────────────────────────────────


def test_F_bang_thuong_khong_regression() -> None:
    t = _table(
        [["Doanh thu thuần", "1.000", "900"], ["Giá vốn hàng bán", "600", "500"]],
        ["Chỉ tiêu", "2017", "2016"],
    )
    long = _std(t)
    assert len(long) == 4
    dt = long[(long["item"] == "Doanh thu thuần") & (long["year"] == 2017)]
    assert dt["value"].iloc[0] == pytest.approx(1000)
    assert dt["metric"].iloc[0] == "net_revenue"


def test_F2_schema_co_du_cot_moi() -> None:
    for col in ("row_idx", "column_label", "period", "source_date", "item_code"):
        assert col in STANDARD_COLUMNS


def test_F3_bang_rong_tra_ve_schema_dung() -> None:
    t = ExtractedTable(doc_id="D", position=1)
    t.df = pd.DataFrame()
    out = SchemaStandardizer().standardize(t)
    assert list(out.columns) == list(STANDARD_COLUMNS)


# ── G. Prompt expose du thong tin ─────────────────────────


def test_G_prompt_liet_ke_item_code_cho_item_mo_ho() -> None:
    df = pd.DataFrame({
        "item": ["- Nguyên giá", "- Nguyên giá"],
        "item_code": ["222", "228"],
        "year": [2017, 2017],
        "period": ["closing", "closing"],
        "column_label": ["31/12/2017", "31/12/2017"],
        "value": [2.3e12, 7.8e10],
    })
    out = pt.format_schema("df1", df)
    assert "CHÚ Ý" in out
    assert "item_code" in out
    assert "222" in out and "228" in out


def test_G2_prompt_liet_ke_column_label_cho_bang_category() -> None:
    df = pd.DataFrame({
        "item": ["Nguyên giá"] * 3,
        "item_code": [None] * 3,
        "year": [2017] * 3,
        "period": ["unknown"] * 3,
        "column_label": ["Nhà cửa", "Máy móc", "Phương tiện"],
        "value": [100.0, 200.0, 300.0],
    })
    out = pt.format_schema("df1", df)
    assert "column_label" in out
    assert "Máy móc" in out
    assert "CHÚ Ý" in out


def test_G3_bang_khong_mo_ho_thi_khong_them_canh_bao() -> None:
    df = pd.DataFrame({
        "item": ["Doanh thu thuần", "Giá vốn"],
        "item_code": ["01", "11"],
        "year": [2017, 2017],
        "period": ["annual", "annual"],
        "column_label": ["2017", "2017"],
        "value": [1000.0, 600.0],
    })
    assert "CHÚ Ý" not in pt.format_schema("df1", df)


# ── card ──────────────────────────────────────────────────


def test_card_co_nhan_category() -> None:
    df = pd.DataFrame({
        "item": ["Nguyên giá"] * 3,
        "year": [2017] * 3,
        "period": ["unknown"] * 3,
        "column_label": ["Nhà cửa vật kiến trúc", "Máy móc thiết bị", "Phương tiện vận tải"],
        "value": [1.0, 2.0, 3.0],
    })
    card = TableCardBuilder().build(doc_id="D", position=1, df=df)
    assert "Máy móc thiết bị" in card


def test_card_khong_lap_lai_nhan_ngay() -> None:
    df = pd.DataFrame({
        "item": ["Tiền", "Tiền"],
        "year": [2017, 2016],
        "period": ["closing", "opening"],
        "column_label": ["31/12/2017", "01/01/2017"],
        "value": [1.0, 2.0],
    })
    card = TableCardBuilder().build(doc_id="D", position=1, df=df)
    assert "Các cột số liệu" not in card
    assert "2017 (closing)" in card


def test_card_khong_chua_row_idx() -> None:
    df = pd.DataFrame({
        "item": ["A"], "year": [2017], "period": ["closing"],
        "row_idx": [0], "column_label": ["Nhà cửa"], "value": [1.0],
    })
    assert "row_idx" not in TableCardBuilder().build(doc_id="D", position=1, df=df)
