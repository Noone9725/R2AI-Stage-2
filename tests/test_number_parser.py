"""Test parse so kieu Viet Nam — nguon sai so lon nhat cua pipeline.

Nham dau phan cach nghin/thap phan lam lech ket qua 1000 lan, va diem
Answer Accuracy mat sach. Day la module dang duoc test ky nhat.
"""

from __future__ import annotations

import pytest

from src.normalization.number_parser import UNIT_SCALES, detect_unit, parse_vn_number


class TestVietnameseFormat:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.234.567", 1234567.0),        # dau . la phan cach nghin
            ("1.234.567,89", 1234567.89),   # dau , la thap phan
            ("1.234", 1234.0),              # 3 chu so sau dau -> nghin
            ("12,5", 12.5),                 # 1 chu so sau dau -> thap phan
            ("0,25", 0.25),
            ("1.000", 1000.0),
            ("123", 123.0),
        ],
    )
    def test_vn_separators(self, raw: str, expected: float) -> None:
        assert parse_vn_number(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1,234,567.89", 1234567.89),   # dang Anh
            ("1,234,567", 1234567.0),
            ("12.5", 12.5),
        ],
    )
    def test_en_separators(self, raw: str, expected: float) -> None:
        assert parse_vn_number(raw) == pytest.approx(expected)

    def test_last_separator_wins(self) -> None:
        """Dau xuat hien sau cung la dau thap phan."""
        assert parse_vn_number("1.234,56") == pytest.approx(1234.56)
        assert parse_vn_number("1,234.56") == pytest.approx(1234.56)


class TestAccountingNegatives:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("(1.234)", -1234.0),           # quy uoc ke toan
            ("( 1.234 )", -1234.0),
            ("(1.234,5)", -1234.5),
            ("-1.234", -1234.0),
            ("(-1.234)", 1234.0),           # hai lan am -> duong
            ("+1.234", 1234.0),
        ],
    )
    def test_negative(self, raw: str, expected: float) -> None:
        assert parse_vn_number(raw) == pytest.approx(expected)


class TestNullCells:
    @pytest.mark.parametrize("raw", ["", "-", "--", "n/a", "N/A", "none", "...", "  ", None])
    def test_empty_is_none(self, raw: object) -> None:
        assert parse_vn_number(raw) is None

    @pytest.mark.parametrize("raw", ["Tài sản ngắn hạn", "abc", "TỔNG CỘNG", "1a2b"])
    def test_text_is_none(self, raw: str) -> None:
        assert parse_vn_number(raw) is None


class TestPercent:
    def test_percent_plain(self) -> None:
        assert parse_vn_number("12,5%") == pytest.approx(12.5)

    def test_percent_as_ratio(self) -> None:
        assert parse_vn_number("12,5%", as_ratio=True) == pytest.approx(0.125)

    def test_negative_percent(self) -> None:
        assert parse_vn_number("(12,5%)", as_ratio=True) == pytest.approx(-0.125)


class TestNumericPassthrough:
    def test_int_and_float(self) -> None:
        assert parse_vn_number(1234) == 1234.0
        assert parse_vn_number(12.5) == pytest.approx(12.5)

    def test_bool_is_not_number(self) -> None:
        """True/False khong bao gio la gia tri o hop le."""
        assert parse_vn_number(True) is None


class TestScale:
    def test_scale_applied(self) -> None:
        assert parse_vn_number("1.234", scale=1e6) == pytest.approx(1.234e9)

    def test_scale_on_native_number(self) -> None:
        assert parse_vn_number(5, scale=1e3) == pytest.approx(5000.0)

    def test_scale_not_applied_to_none(self) -> None:
        assert parse_vn_number("-", scale=1e6) is None


class TestDetectUnit:
    @pytest.mark.parametrize(
        "context,label",
        [
            ("Đơn vị: triệu đồng", "trieu dong"),
            ("Đơn vị: tỷ đồng", "ty dong"),
            ("Đơn vị: nghìn đồng", "nghin dong"),
            ("Đơn vị: ngàn đồng", "nghin dong"),
            ("Unit: million VND", "million"),
            ("Đơn vị: VND", "vnd"),
        ],
    )
    def test_detect(self, context: str, label: str) -> None:
        found, scale = detect_unit(context)
        assert found == label
        assert scale == UNIT_SCALES[label]

    def test_no_unit(self) -> None:
        assert detect_unit("Bảng cân đối kế toán") == (None, 1.0)

    def test_bigger_unit_wins(self) -> None:
        """'tỷ đồng' phai thang 'đồng' — thu tu pattern quan trong."""
        label, scale = detect_unit("Đơn vị tính: tỷ đồng")
        assert (label, scale) == ("ty dong", 1e9)

    def test_scale_roundtrip(self) -> None:
        """Ket hop detect_unit + parse: 1.234 trieu dong = 1.234e9 VND."""
        _, scale = detect_unit("Đơn vị: triệu đồng")
        assert parse_vn_number("1.234", scale=scale) == pytest.approx(1.234e9)
