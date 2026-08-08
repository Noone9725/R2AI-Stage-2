"""Test trich xuat bang HTML — dinh dang THUC TE cua kho ViFinQA.

Cac fixture trong file nay la doan CAT NGUYEN VAN tu
    data/raw/financial_statements/AAA/2020/
        AAA_financial_statements_2020_consolidated/..._extracted.txt
nen chung khoa dung hanh vi tren du lieu that, khong phai gia dinh.
"""

from __future__ import annotations

from src.extraction.html_table import (
    count_html_tables,
    extract_html_tables,
    split_glued_numbers,
)
from src.extraction.table_detector import TableDetector
from src.schemas import RawDocument

# Doan that: bang can doi ke toan AAA 2020, giu nguyen loi OCR
# ("Másố", "BẢNG CẦN ĐỐI") va cac o rowspan bi dan so.
REAL_TXT = """===== PAGE 8 =====
Công ty Cổ phần Nhựa An Phát Xanh

B01-DN/HN

BẢNG CẦN ĐỐI KẾ TOẢN HỢP NHẤT

ngày 31 tháng 12 năm 2020

Đơn vị tính: VND

<table><tr><td>Másố</td><td>TÀI SẢN</td><td>Thuyếtminh</td><td>Số cuối năm</td><td>Số đầu năm</td></tr>\
<tr><td>100</td><td>A. TÀI SẢN NGĂN HẠN</td><td></td><td>4.496.050.828.524</td><td>4.971.363.590.401</td></tr>\
<tr><td>110</td><td>I. Tiền và các khoản tương đương tiền</td><td rowspan="3">5</td>\
<td rowspan="3">963.717.122.052237.314.356.418726.402.765.634</td>\
<td rowspan="3">291.674.680.985233.349.201.53558.325.479.450</td></tr>\
<tr><td>111</td><td>1. Tiền</td></tr>\
<tr><td>112</td><td>2. Các khoản tương đương tiền</td></tr>\
<tr><td>120</td><td>II. Đầu tư tài chính ngắn hạn</td><td rowspan="3">6</td>\
<td rowspan="2">758.600.000.000-</td><td rowspan="2">1.251.822.102.19259.670.020.000</td></tr>\
<tr><td>121</td><td>1. Chứng khoán kinh doanh</td></tr>\
<tr><td>123</td><td>2. Đầu tư nằm giữ đến ngày đáo hạn</td><td>758.600.000.000</td>\
<td>1.192.152.082.192</td></tr></table>
"""


def _doc(text: str) -> RawDocument:
    doc_id = "AAA_financial_statements_2020_consolidated"
    return RawDocument(doc_id=doc_id, path=f"raw/{doc_id}.txt", text=text)


# ── tach so bi dan ────────────────────────────────────────


def test_split_glued_numbers_tach_dung_ba_gia_tri():
    """Quy uoc nhom 3 chu so lam phep tach duy nhat."""
    assert split_glued_numbers(
        "963.717.122.052237.314.356.418726.402.765.634"
    ) == ["963.717.122.052", "237.314.356.418", "726.402.765.634"]


def test_split_glued_numbers_giu_o_trong_dang_gach_ngang():
    """"-" la o RONG trong BCTC, khong phai dau tru cua so ke tiep."""
    assert split_glued_numbers("758.600.000.000-") == ["758.600.000.000", "-"]


def test_split_glued_numbers_khong_cat_van_ban():
    assert split_glued_numbers("Hàng tồn kho") == ["Hàng tồn kho"]
    assert split_glued_numbers("7.1") == ["7.1"]
    assert split_glued_numbers("(11.746.705.700)") == ["(11.746.705.700)"]


# ── giai ma bang ──────────────────────────────────────────


def test_extract_tra_ve_luoi_chu_nhat():
    tables = extract_html_tables(REAL_TXT)
    assert len(tables) == 1
    rows = tables[0].rows
    assert all(len(r) == 5 for r in rows), "moi dong phai cung so cot"


def test_rowspan_duoc_phan_phoi_dung_tung_dong():
    """Ba gia tri dan lien phai tra ve dung dong 110/111/112."""
    rows = extract_html_tables(REAL_TXT)[0].rows
    by_code = {r[0]: r for r in rows}
    assert by_code["110"][3] == "963.717.122.052"
    assert by_code["111"][3] == "237.314.356.418"
    assert by_code["112"][3] == "726.402.765.634"
    assert by_code["110"][4] == "291.674.680.985"
    assert by_code["112"][4] == "58.325.479.450"


def test_o_gop_khong_phai_so_thi_lap_xuong_duoi():
    """So hieu thuyet minh rowspan=3 ap dung cho ca ba dong."""
    rows = extract_html_tables(REAL_TXT)[0].rows
    by_code = {r[0]: r for r in rows}
    assert by_code["110"][2] == "5"
    assert by_code["111"][2] == "5"
    assert by_code["112"][2] == "5"


def test_so_lieu_tai_lap_dung_theo_phep_cong_bctc():
    """Kiem tra doc lap: 121 + 123 phai bang chi tieu me 120."""
    rows = extract_html_tables(REAL_TXT)[0].rows
    by_code = {r[0]: r for r in rows}

    def num(s: str) -> int:
        return 0 if s.strip() in {"", "-"} else int(s.replace(".", ""))

    assert num(by_code["121"][4]) + num(by_code["123"][4]) == num(by_code["120"][4])


def test_page_marker_duoc_gan_vao_bang():
    assert extract_html_tables(REAL_TXT)[0].page == 8


def test_position_dem_tu_1():
    assert extract_html_tables(REAL_TXT)[0].position == 1


def test_count_html_tables():
    assert count_html_tables(REAL_TXT) == 1


# ── tich hop voi TableDetector ────────────────────────────


def test_detector_uu_tien_duong_html():
    tables = TableDetector().detect(_doc(REAL_TXT))
    assert len(tables) == 1
    assert tables[0].rows[0][1] == "TÀI SẢN"


def test_detector_gan_section_du_ocr_sai_dau():
    """OCR ghi "CẦN ĐỐI"/"TOẢN" thay vi "CÂN ĐỐI"/"TOÁN" — van phai nhan ra."""
    assert TableDetector().detect(_doc(REAL_TXT))[0].section == "balance_sheet"


def test_detector_lay_tieu_de_khong_phai_dong_don_vi():
    title = TableDetector().detect(_doc(REAL_TXT))[0].title
    assert "ĐỐI KẾ TO" in title
    assert "Đơn vị tính" not in title


def test_context_before_khong_chua_markup():
    ctx = TableDetector().detect(_doc(REAL_TXT))[0].context_before
    assert "<td>" not in ctx and "<table" not in ctx
    assert "Đơn vị tính: VND" in ctx, "context phai giu don vi cho normalization"


def test_table_ref_dung_dinh_dang_submission():
    table = TableDetector().detect(_doc(REAL_TXT))[0]
    assert table.table_ref == "AAA_financial_statements_2020_consolidated|1"


def test_bang_ngan_hang_dung_ten_bao_cao_tinh_hinh_tai_chinh():
    """Thong tu 49/2014/TT-NHNN goi bang can doi la "tinh hinh tai chinh"."""
    text = (
        "===== PAGE 5 =====\n"
        "BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHÀT\n"
        "tại ngày 31 tháng 12 năm 2024\n"
        "<table><tr><td>Chỉ tiêu</td><td>Số cuối năm</td></tr>"
        "<tr><td>Tiền mặt</td><td>1.234.567.890</td></tr></table>\n"
    )
    assert TableDetector().detect(_doc(text))[0].section == "balance_sheet"
