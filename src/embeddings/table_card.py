"""Bang -> 'card' text de embed/BM25.

Bang so lieu embed truc tiep rat kem (toan chu so). Card la ban dich bang
thanh mot doan van ban giau tin hieu: cong ty, nam, loai bao cao, tieu de,
danh sach chi tieu. Chat luong card quyet dinh phan lon diem F2.
"""

from __future__ import annotations

import re

import pandas as pd

# Nhan cot chi la ngay/nam — da duoc bieu dien o dong "Cac nam co so lieu".
_DATE_LIKE_RE = re.compile(r"^\s*\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{4}|^\s*\d{4}\s*$")

_SECTION_LABELS: dict[str, str] = {
    "balance_sheet": "Bảng cân đối kế toán",
    "income_statement": "Báo cáo kết quả hoạt động kinh doanh",
    "cash_flow": "Báo cáo lưu chuyển tiền tệ",
    "notes": "Thuyết minh báo cáo tài chính",
}

_REPORT_LABELS: dict[str, str] = {
    "consolidated": "hợp nhất",
    "separate": "riêng lẻ",
}


class TableCardBuilder:
    """Sinh card tu manifest entry + DataFrame chuan hoa."""

    def __init__(self, max_items: int = 30, max_metrics: int = 15,
                 max_categories: int = 12):
        self.max_items = max_items
        self.max_metrics = max_metrics
        self.max_categories = max_categories

    def build(
        self,
        *,
        doc_id: str,
        position: int,
        title: str = "",
        section: str | None = None,
        ticker: str | None = None,
        year: int | None = None,
        report_type: str | None = None,
        unit: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> str:
        parts: list[str] = []

        if ticker:
            parts.append(f"Công ty: {ticker}")
        if year:
            parts.append(f"Năm: {year}")
        if report_type and report_type in _REPORT_LABELS:
            parts.append(f"Loại báo cáo: {_REPORT_LABELS[report_type]}")
        if section:
            parts.append(f"Loại bảng: {_SECTION_LABELS.get(section, section)}")
        if title:
            parts.append(f"Tiêu đề: {title}")
        if unit:
            parts.append(f"Đơn vị: {unit}")

        if df is not None and not df.empty:
            parts.extend(self._from_df(df))

        parts.append(f"Mã bảng: {doc_id}|{position}")
        return "\n".join(parts)

    def _from_df(self, df: pd.DataFrame) -> list[str]:
        out: list[str] = []

        if "item" in df.columns:
            items = (
                df["item"].dropna().astype(str).str.strip()
                .loc[lambda s: s != ""].drop_duplicates().head(self.max_items).tolist()
            )
            if items:
                out.append("Các chỉ tiêu: " + "; ".join(items))

        if "metric" in df.columns:
            metrics = (
                df["metric"].dropna().astype(str).drop_duplicates()
                .head(self.max_metrics).tolist()
            )
            if metrics:
                out.append("Chỉ tiêu chuẩn hoá: " + ", ".join(metrics))

        if "year" in df.columns:
            out.append("Các năm có số liệu: " + self._years_text(df))

        out.extend(self._categories_text(df))

        return out

    def _categories_text(self, df: pd.DataFrame) -> list[str]:
        """Nhan cot category — mang ngu nghia retrieval that.

        Bang thuyet minh TSCD co truc cot la loai tai san ("Nha cua vat kien
        truc", "May moc thiet bi"). Cau hoi ve "nguyen gia may moc thiet bi"
        chi khop duoc bang nay neu card chua cac nhan do.

        KHONG dua `row_idx` vao card — no la so thu tu ky thuat, khong co
        gia tri retrieval nao.
        """
        if "column_label" not in df.columns:
            return []
        labels = [
            s for s in df["column_label"].dropna().astype(str).unique().tolist()
            if s.strip()
        ]
        # Bang truc thoi gian co column_label = "31/12/2015" — da co o dong
        # "Cac nam co so lieu", nhac lai chi lam loang card.
        labels = [s for s in labels if not _DATE_LIKE_RE.match(s)]
        if not labels:
            return []
        return ["Các cột số liệu: " + " | ".join(labels[: self.max_categories])]

    @staticmethod
    def _years_text(df: pd.DataFrame) -> str:
        """Nam kem period, gon. Khong duoc ghi "2015, 2015".

        Bang co ca 31/12/2015 va 01/01/2016 deu tro ve nam 2015 (dung ve ke
        toan). Ghi "2015" hai lan lam card nhieu va lam retrieval hieu sai
        do phu du lieu; ghi "2015 (closing, opening)" thi vua dung vua ngan.
        """
        pairs: dict[int, list[str]] = {}
        has_period = "period" in df.columns
        for _, row in df[["year"] + (["period"] if has_period else [])].dropna(
            subset=["year"]
        ).iterrows():
            year = int(row["year"])
            per = str(row["period"]) if has_period and pd.notna(row.get("period")) else ""
            bucket = pairs.setdefault(year, [])
            if per and per not in bucket and per != "unknown":
                bucket.append(per)

        parts: list[str] = []
        for year in sorted(pairs):
            periods = pairs[year]
            parts.append(f"{year} ({', '.join(periods)})" if periods else str(year))
        return ", ".join(parts)
