"""Test don vi dap an — Phase 1 fix.

Bug goc: pipeline chuan hoa moi gia tri ve VND nhung 620/1012 cau hoi lai
hoi bang ty/trieu/nghin dong -> dap an lech 1e3..1e9 lan.
"""

from __future__ import annotations

import pytest

from src.normalization.answer_unit import (
    AnswerNormalizer,
    AskedUnit,
    detect_asked_unit,
    divisor_for,
)


# ── detect_asked_unit ─────────────────────────────────────


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023 là bao nhiêu tỷ đồng?",
         AskedUnit.TY_DONG),
        ("Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?",
         AskedUnit.TRIEU_DONG),
        ("Vay dài hạn của công ty mẹ HNG là bao nhiêu nghìn đồng?",
         AskedUnit.NGHIN_DONG),
        ("Số dư trả trước cho người bán của VPI cuối năm 2024 tính bằng đồng?",
         AskedUnit.DONG),
        ("Tỷ lệ biểu quyết của Visorutex của công ty mẹ GVR là bao nhiêu %?",
         AskedUnit.PERCENT),
        ("ROE của VNM năm 2020 là bao nhiêu phần trăm?", AskedUnit.PERCENT),
        ("Hệ số khả năng thanh toán lãi vay của KBC là bao nhiêu lần?",
         AskedUnit.TIMES),
        ("Tổng tỷ lệ quyền biểu quyết của Đèo Cả năm 2023 là bao nhiêu?",
         AskedUnit.NONE),
    ],
)
def test_detect_asked_unit(question: str, expected: AskedUnit) -> None:
    assert detect_asked_unit(question) == expected


def test_khong_dau_van_nhan_dien_duoc() -> None:
    """Cau hoi go khong dau van phai ra dung don vi."""
    assert detect_asked_unit("LNST nam 2023 la bao nhieu ty dong?") == AskedUnit.TY_DONG
    assert detect_asked_unit("Lai tien gui la bao nhieu trieu dong?") == AskedUnit.TRIEU_DONG


def test_lan_uu_tien_hon_percent() -> None:
    """7 cau hoi that chua CA '%' lan 'lan'. Don vi DAP AN la 'lan'."""
    q = ("Trong giai đoạn 2016-2020, vào năm KBC có tỷ số D/E cao nhất (tính theo %), "
         "hệ số khả năng thanh toán lãi vay là bao nhiêu lần?")
    assert detect_asked_unit(q) == AskedUnit.TIMES


def test_ty_dong_uu_tien_hon_dong() -> None:
    """'ty dong' chua chuoi 'dong' — khong duoc khop DONG truoc."""
    assert detect_asked_unit("... bao nhiêu tỷ đồng?") == AskedUnit.TY_DONG
    assert detect_asked_unit("... bao nhiêu triệu đồng?") == AskedUnit.TRIEU_DONG
    assert detect_asked_unit("... bao nhiêu nghìn đồng?") == AskedUnit.NGHIN_DONG


# ── conversion (cac case bat buoc theo yeu cau) ───────────


@pytest.mark.parametrize(
    "raw, unit, expected",
    [
        (1_000_000_000.0, AskedUnit.TY_DONG, 1.0),
        (1_500_000.0, AskedUnit.TRIEU_DONG, 1.5),
        (250_000.0, AskedUnit.NGHIN_DONG, 250.0),
        (150.0, AskedUnit.DONG, 150.0),
        (462_000_000_000.0, AskedUnit.TY_DONG, 462.0),
        (1_071_561_008_455.0, AskedUnit.TY_DONG, 1071.56),
    ],
)
def test_monetary_conversion(raw: float, unit: AskedUnit, expected: float) -> None:
    out = AnswerNormalizer().apply(raw, unit)
    assert out.value == pytest.approx(expected)
    assert out.raw_value == raw


def test_percent_khong_nhan_100_lan_nua() -> None:
    """Prompt (rule 6) da yeu cau code tra 15.3 chu khong phai 0.153.
    Normalizer KHONG duoc nhan 100 -> se thanh 1530."""
    out = AnswerNormalizer().apply(15.3, AskedUnit.PERCENT)
    assert out.value == pytest.approx(15.3)
    assert out.converted is False


def test_lan_giu_nguyen_khong_nhan_100() -> None:
    """1.53 lan phai ra 1.53, KHONG ra 153."""
    out = AnswerNormalizer().apply(1.53, AskedUnit.TIMES)
    assert out.value == pytest.approx(1.53)
    assert out.divisor == 1.0


def test_percent_va_lan_dung_conversion_khac_nhau() -> None:
    """Cung mot ratio, hai don vi -> KHONG duoc cho ket qua giong nhau
    theo cach gop chung mot cong thuc."""
    n = AnswerNormalizer()
    assert n.apply(0.153, AskedUnit.PERCENT).unit is AskedUnit.PERCENT
    assert n.apply(1.53, AskedUnit.TIMES).unit is AskedUnit.TIMES
    # ca hai deu khong bi chia cho 1e9 nhu tien te
    assert n.apply(0.153, AskedUnit.PERCENT).divisor == 1.0
    assert n.apply(1.53, AskedUnit.TIMES).divisor == 1.0


def test_none_giu_nguyen_gia_tri() -> None:
    out = AnswerNormalizer().apply(12345.678, AskedUnit.NONE)
    assert out.value == pytest.approx(12345.68)
    assert out.converted is False


def test_so_am_van_doi_dung() -> None:
    """Lo/chi phi trong BCTC la so am — khong duoc mat dau."""
    out = AnswerNormalizer().apply(-2_500_000_000.0, AskedUnit.TY_DONG)
    assert out.value == pytest.approx(-2.5)


def test_normalize_tu_question_text() -> None:
    """Duong dung that: chi truyen value + question."""
    out = AnswerNormalizer().normalize(
        462_000_000_000.0, "LNST của CTCP Chứng khoán FPT năm 2023 là bao nhiêu tỷ đồng?"
    )
    assert out.value == pytest.approx(462.0)
    assert out.unit is AskedUnit.TY_DONG
    assert out.converted is True


def test_divisor_for() -> None:
    assert divisor_for("... bao nhiêu tỷ đồng?") == 1e9
    assert divisor_for("... bao nhiêu triệu đồng?") == 1e6
    assert divisor_for("... bao nhiêu nghìn đồng?") == 1e3
    assert divisor_for("... bao nhiêu lần?") == 1.0
    assert divisor_for("... bao nhiêu %?") == 1.0
