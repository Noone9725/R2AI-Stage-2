"""Anh xa ten cong ty -> ticker, va viec loc ma gia trong QueryAnalyzer.

Cac ca kiem tra o day KHONG phai gia dinh: moi cai deu tuong ung mot cau
hoi that trong `data/questions/questions.jsonl` da tung bi trich sai.

Dung CSV rieng cho tung test thay vi `data/questions/code_stock.csv` de
test khong phu thuoc du lieu tai ve, nhung noi dung lay dung tu file that.
"""

from __future__ import annotations

import csv

import pytest

from src.retrieval.company_map import CompanyMap
from src.retrieval.query_analyzer import QueryAnalyzer

# Trich tu code_stock.csv that. Gom du cac ca kho: hai ngan hang chung
# tien to "Sai Gon", ten mot tu trong ten phap ly dai, ma co chu so.
_ROWS = [
    ("SHB", "Ngân hàng TMCP Sài Gòn - Hà Nội"),
    ("SGB", "Ngân hàng TMCP Sài Gòn Công Thương"),
    ("ACB", "Ngân hàng TMCP Á Châu"),
    ("MBB", "Ngân hàng TMCP Quân đội"),
    ("EIB", "Ngân hàng TMCP Xuất nhập khẩu Việt Nam"),
    ("HPG", "CTCP Tập đoàn Hòa Phát"),
    ("HSG", "CTCP Tập đoàn Hoa Sen"),
    ("NKG", "CTCP Thép Nam Kim"),
    ("VNM", "CTCP Sữa Việt Nam"),
    ("MSN", "CTCP Tập đoàn Masan"),
    ("GEX", "CTCP Tập đoàn GELEX"),
    ("KBC", "Tổng Công ty Phát triển Đô thị Kinh Bắc - CTCP"),
    ("HT1", "CTCP Xi Măng Vicem Hà Tiên"),
    ("PC1", "CTCP Tập Đoàn PC1"),
    ("PNJ", "CTCP Vàng bạc Đá quý Phú Nhuận"),
    ("SNZ", "Tổng CTCP Phát triển Khu Công nghiệp"),
]


@pytest.fixture(scope="module")
def cmap(tmp_path_factory: pytest.TempPathFactory) -> CompanyMap:
    path = tmp_path_factory.mktemp("q") / "code_stock.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Mã CK", "Tên công ty"])
        w.writerows(_ROWS)
    return CompanyMap(path)


@pytest.fixture(scope="module")
def qa(cmap: CompanyMap) -> QueryAnalyzer:
    return QueryAnalyzer(company_map=cmap)


def test_loads_every_ticker(cmap: CompanyMap) -> None:
    assert cmap.tickers == {t for t, _ in _ROWS}


def test_full_legal_name(cmap: CompanyMap) -> None:
    assert cmap.resolve("Ngân hàng TMCP Á Châu năm 2016") == ["ACB"]


def test_accent_insensitive(cmap: CompanyMap) -> None:
    """OCR va cau hoi khac dau nhau — "Hoà" vs "Hòa" phai cung khop."""
    assert cmap.resolve("Hoà Phát") == cmap.resolve("Hòa Phát") == ["HPG"]


def test_legal_form_ignored(cmap: CompanyMap) -> None:
    """"CTCP" va "Cong ty Co phan" la hai cach viet cua cung doanh nghiep."""
    assert cmap.resolve("Công ty Cổ phần Tập đoàn Hòa Phát") == ["HPG"]


def test_shared_prefix_picks_longest(cmap: CompanyMap) -> None:
    """SHB vs SGB dung chung tien to "Ngan hang TMCP Sai Gon"."""
    assert cmap.resolve("Ngân hàng TMCP Sài Gòn - Hà Nội") == ["SHB"]
    assert cmap.resolve("Ngân hàng TMCP Sài Gòn Công Thương") == ["SGB"]


def test_multiple_companies_one_question(cmap: CompanyMap) -> None:
    got = cmap.resolve("CTCP Tập đoàn Hòa Phát và Ngân hàng TMCP Quân đội")
    assert sorted(got) == ["HPG", "MBB"]


def test_short_trading_names(cmap: CompanyMap) -> None:
    """Cau 384: nhom nay duoc goi bang ten thuong goi, khong phai ten phap ly."""
    got = cmap.resolve("Trong nhóm Hoà Phát, Hoa Sen và Nam Kim")
    assert sorted(got) == ["HPG", "HSG", "NKG"]


def test_brand_aliases(cmap: CompanyMap) -> None:
    """Ten thuong mai khong suy ra duoc tu ten phap ly (cau 792, 412)."""
    assert sorted(cmap.resolve("MBBank lớn hơn Eximbank")) == ["EIB", "MBB"]
    assert cmap.resolve("Vinamilk") == ["VNM"]
    assert cmap.resolve("GELEX") == ["GEX"]


def test_single_word_suffix_is_not_an_alias(cmap: CompanyMap) -> None:
    """Chan hoi quy: hau to mot tu la tu tieng Viet thong thuong.

    Cho phep chung lam alias khien HT1 khop 203 cau hoi va PNJ khop 179
    cau khong lien quan, vi "tien" / "nhuan" xuat hien khap bo cau hoi.
    """
    assert cmap.resolve("số dư tiền và tương đương tiền cuối năm") == []
    assert cmap.resolve("lợi nhuận sau thuế tăng") == []
    assert cmap.resolve("khu công nghiệp") == []


# ── QueryAnalyzer ─────────────────────────────────────────


def test_digit_suffix_ticker(qa: QueryAnalyzer) -> None:
    """HT1 / PC1: chi 2 chu cai + so, regex 3-4 chu cai truot (cau 276, 283)."""
    assert qa.extract_tickers("Số dư vay ngắn hạn của HT1 cuối năm 2020") == ["HT1"]
    got = qa.extract_tickers("công ty mẹ PC1 chênh lệch so với GELEX")
    assert sorted(got) == ["GEX", "PC1"]


def test_lowercase_ticker_list(qa: QueryAnalyzer) -> None:
    """Cau 431 liet ke ma bang chu thuong."""
    got = qa.extract_tickers("ngành BĐS (gồm các công ty hpg,kbc,nkg)")
    assert sorted(got) == ["HPG", "KBC", "NKG"]


@pytest.mark.parametrize(
    "text",
    [
        "có CFO dương trong cả hai năm 2020 và 2021",       # CFO
        "Thuế TNDN phải nộp cuối năm 2024",                  # TNDN
        "tỉ số thanh toán nhanh và LNTT dương",              # LNTT
        "tỷ lệ cho vay trên tổng tiền gửi (LDR)",            # LDR
        "GTCG nhận thế chấp, cầm cố",                        # GTCG
        "tài sản bộ phận BOT trên tổng tài sản",             # BOT
        "Ngân hàng TMCP nào có ROE cao nhất",                # TMCP, ROE
    ],
)
def test_financial_acronyms_are_not_tickers(qa: QueryAnalyzer, text: str) -> None:
    """Viet tat tai chinh viet hoa dung dang ma CK.

    Chung khong nam trong `code_stock.csv`, va kho tai lieu chi co dung
    100 thu muc ticker trung khop CSV do — nhan chung chi lam ban filter.
    """
    assert qa.extract_tickers(text) == []


def test_ticker_in_parentheses_wins(qa: QueryAnalyzer) -> None:
    assert qa.extract_tickers("Công ty CP Sữa Việt Nam (VNM) năm 2016") == ["VNM"]


def test_missing_csv_degrades_quietly(tmp_path) -> None:
    """Thieu file thi tra ve rong, khong nem loi — pipeline van chay duoc."""
    cm = CompanyMap(tmp_path / "khong-ton-tai.csv")
    assert len(cm) == 0
    assert cm.tickers == set()
    assert cm.resolve("Ngân hàng TMCP Á Châu") == []
