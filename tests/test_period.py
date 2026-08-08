"""Test giai ma year/period tu header BCTC — Phase 2.

Bug goc: '31/12/2015' va '01/01/2015' deu -> year=2015, tao hai dong cung
(item, year) voi gia tri khac nhau. Do tren 3.000 bang: 4.470 collision.
"""

from __future__ import annotations

import pytest

from src.normalization.period import (
    Period,
    detect_requested_period,
    find_period_cols,
    parse_column_period,
)


# ── basic year ────────────────────────────────────────────


def test_bare_year() -> None:
    p = parse_column_period("2015")
    assert (p.year, p.period) == (2015, Period.ANNUAL)


def test_year_co_tien_to_chu() -> None:
    p = parse_column_period("Năm 2015")
    assert (p.year, p.period) == (2015, Period.ANNUAL)


# ── closing ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "header", ["31/12/2015", "31/12/2015 VND", "31/12/2015VND", "31.12.2015Triệu đồng",
               "Tại ngày 31/12/2015", "31-12-2015"],
)
def test_closing_31_12(header: str) -> None:
    p = parse_column_period(header)
    assert p is not None, header
    assert (p.year, p.period) == (2015, Period.CLOSING)


# ── opening (lui mot nam) ─────────────────────────────────


@pytest.mark.parametrize(
    "header", ["01/01/2015", "1/1/2015", "1/1/2015 VND", "Tại ngày1.1.2015Triệu đồng"],
)
def test_opening_01_01_lui_mot_nam(header: str) -> None:
    """So du dau ky 01/01/2015 == so du cuoi ky 31/12/2014."""
    p = parse_column_period(header)
    assert p is not None, header
    assert p.year == 2014, "phai lui ve nam truoc"
    assert p.period is Period.OPENING


# ── other date: KHONG lui nam ─────────────────────────────


@pytest.mark.parametrize("header", ["30/03/2015", "8/8/2015", "15/10/2015"])
def test_other_date_giu_nguyen_nam(header: str) -> None:
    p = parse_column_period(header)
    assert p is not None, header
    assert p.year == 2015, "ngay le KHONG duoc lui nam"
    assert p.period is Period.POINT_IN_TIME


# ── multiple columns: khong collision ─────────────────────


def test_31_12_va_01_01_cung_nam_khong_collision() -> None:
    """Chinh la bug goc."""
    cols = ["31/12/2015", "01/01/2015"]
    out = find_period_cols(cols)
    keys = {(p.year, p.period) for p in out.values()}
    assert len(keys) == 2, "hai cot phai ra hai semantic key khac nhau"
    assert out["31/12/2015"].year == 2015
    assert out["01/01/2015"].year == 2014


def test_multi_year_mapping_nhat_quan() -> None:
    """31/12/2016 | 01/01/2016 | 31/12/2015
    -> 01/01/2016 va 31/12/2015 cung tro ve nam 2015 (dung ve ke toan),
    nhung period khac nhau nen van phan biet duoc."""
    out = find_period_cols(["31/12/2016", "01/01/2016", "31/12/2015"])
    assert out["31/12/2016"].year == 2016
    assert out["01/01/2016"].year == 2015
    assert out["31/12/2015"].year == 2015
    assert out["01/01/2016"].period is Period.OPENING
    assert out["31/12/2015"].period is Period.CLOSING
    assert len({(p.year, p.period) for p in out.values()}) == 3


def test_cot_trung_lap_hoan_toan_chi_giu_mot() -> None:
    out = find_period_cols(["31/12/2015", "31/12/2015 VND"])
    assert len(out) == 1


def test_exclude_duoc_ton_trong() -> None:
    out = find_period_cols(["Chỉ tiêu", "31/12/2015"], exclude={"Chỉ tiêu"})
    assert list(out) == ["31/12/2015"]


# ── van xuoi KHONG duoc nhan la cot nam ───────────────────


@pytest.mark.parametrize(
    "header",
    [
        "BCTC tổng hợp năm 2025 (Đã kiểm toán)",
        "bổ nhiệm ngày 6 tháng 6 năm 2025",
        "Từ ngày 27 tháng 4 năm 2021đến ngày 31 tháng 8 năm 2021",
        "Mẫu số B09 - DNThông tư 200/2014/TT-BTC ngày 22/12/2014 của Bộ Tài chính",
        "Trường ban(bổ nhiệm ngày 7.4.2018)",
    ],
)
def test_van_xuoi_bi_loai(header: str) -> None:
    """1.417/3.000 cot chua nam la van xuoi — parser cu nhan het."""
    assert parse_column_period(header) is None, header


# ── OCR malformed: khong crash ────────────────────────────


@pytest.mark.parametrize("header", ["45/13/2015", "99/99/2015", "00/00/2015"])
def test_ngay_khong_hop_le_fallback_unknown(header: str) -> None:
    p = parse_column_period(header)
    assert p is not None
    assert p.year == 2015
    assert p.period is Period.UNKNOWN, "fallback ro rang, khong doan bua"


@pytest.mark.parametrize("header", ["", None, "Chỉ tiêu", "Mã số", "VND", "---", 123])
def test_khong_phai_cot_thoi_gian(header: object) -> None:
    assert parse_column_period(header) is None


# ── traceability ──────────────────────────────────────────


def test_giu_raw_header_va_source_date() -> None:
    p = parse_column_period("31/12/2015 VND")
    assert p.raw_header == "31/12/2015 VND"
    assert p.source_date == "31/12/2015"
    assert p.label == "2015 (closing)"


def test_annual_khong_co_source_date() -> None:
    p = parse_column_period("2015")
    assert p.source_date is None


# ── regression: bang chi co nam ───────────────────────────


def test_bang_plain_year_khong_regression() -> None:
    out = find_period_cols(["Chỉ tiêu", "2015", "2014"], exclude={"Chỉ tiêu"})
    assert out["2015"].year == 2015
    assert out["2014"].year == 2014
    assert all(p.period is Period.ANNUAL for p in out.values())


# ── cot thoi diem viet bang CHU ───────────────────────────
# 1.890 cot dang nay tren 50 bao cao — nhieu hon ca cot ghi ngay (1.573).


@pytest.mark.parametrize(
    "header, context_year, expected_year, expected_period",
    [
        ("Số cuối năm", 2022, 2022, Period.CLOSING),
        ("Số dư cuối năm", 2022, 2022, Period.CLOSING),
        ("Số dư cuối kỳ", 2019, 2019, Period.CLOSING),
        ("Số đầu năm", 2022, 2021, Period.OPENING),
        ("Số đầu năm (Trình bày lại)", 2022, 2021, Period.OPENING),
        ("Năm nay", 2020, 2020, Period.ANNUAL),
        ("Năm nay VND", 2020, 2020, Period.ANNUAL),
        ("Năm trước", 2020, 2019, Period.ANNUAL),
        ("Năm trước (Trình bày lại)", 2020, 2019, Period.ANNUAL),
    ],
)
def test_cot_viet_bang_chu(
    header: str, context_year: int, expected_year: int, expected_period: Period
) -> None:
    p = parse_column_period(header, context_year)
    assert p is not None, header
    assert (p.year, p.period) == (expected_year, expected_period)


def test_cot_chu_can_context_year() -> None:
    """Khong biet nam bao cao thi khong doan — tra None thay vi bia."""
    assert parse_column_period("Số cuối năm") is None
    assert parse_column_period("Năm trước") is None


def test_cuoi_nam_va_dau_nam_khong_collision() -> None:
    """Cap cot pho bien nhat trong BCTC ngan hang/CTCK."""
    out = find_period_cols(["Số cuối năm", "Số đầu năm"], context_year=2022)
    assert out["Số cuối năm"].year == 2022
    assert out["Số đầu năm"].year == 2021
    assert len({(p.year, p.period) for p in out.values()}) == 2


def test_nam_nay_nam_truoc_khong_collision() -> None:
    out = find_period_cols(["Năm nay", "Năm trước"], context_year=2020)
    assert {(p.year, p.period.value) for p in out.values()} == {
        (2020, "annual"), (2019, "annual")
    }


def test_ngay_uu_tien_hon_chu() -> None:
    """Cot vua co ngay vua co chu -> ngay la tin hieu chac hon."""
    p = parse_column_period("Số cuối năm 31/12/2018", 2022)
    assert (p.year, p.period) == (2018, Period.CLOSING)


# ── phia cau hoi ──────────────────────────────────────────


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Số dư phải thu của HHV đến ngày 01/01/2022 là bao nhiêu?", Period.OPENING),
        ("Thuế TNDN phải nộp đầu năm 2021 của PC1 là bao nhiêu?", Period.OPENING),
        ("Số dư cho vay của ACB cuối năm 2022 là bao nhiêu?", Period.CLOSING),
        ("Vốn chủ sở hữu của FIT vào ngày 31/12/2015?", Period.CLOSING),
        ("Chi phí dự phòng trong năm 2020 là bao nhiêu?", Period.ANNUAL),
        ("Lãi tiền gửi năm 2018 của VJC là bao nhiêu?", None),
    ],
)
def test_detect_requested_period(question: str, expected: Period | None) -> None:
    assert detect_requested_period(question) == expected


def test_opening_uu_tien_khi_cau_hoi_co_ca_hai() -> None:
    """'tu dau nam den cuoi nam' — OPENING la tin hieu hiem, co chu dich."""
    q = "Mức tăng hàng tồn kho từ đầu năm đến cuối năm 2022 của HPG?"
    assert detect_requested_period(q) is Period.OPENING
