"""Anh xa thuat ngu tai chinh tieng Viet <-> khoa chuan.

Dung cho 2 viec:
  1. Query analysis: cau hoi "Doanh thu thuan..." -> metric "net_revenue".
  2. Table card: gan nhan chi tieu de retrieval khop tot hon.

Tu dien de o `configs/prompts/` khong hop — day la du lieu, khong phai prompt,
nen nhung truc tiep vao code cho de test va version.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.vn_text import normalize_text

# metric_key -> cac cach dien dat trong BCTC/cau hoi (khong dau, lowercase)
FINANCIAL_TERMS: dict[str, tuple[str, ...]] = {
    # ── Income statement ──
    "net_revenue": (
        "doanh thu thuan", "doanh thu thuan ve ban hang",
        "doanh thu thuan ve ban hang va cung cap dich vu", "net revenue", "net sales",
    ),
    "gross_revenue": ("tong doanh thu", "doanh thu ban hang", "gross revenue"),
    "cost_of_goods_sold": ("gia von hang ban", "cost of goods sold", "cogs"),
    "gross_profit": ("loi nhuan gop", "lai gop", "gross profit"),
    "operating_profit": (
        "loi nhuan thuan tu hoat dong kinh doanh", "loi nhuan hoat dong",
        "operating profit", "ebit",
    ),
    "financial_expense": ("chi phi tai chinh", "financial expense"),
    "interest_expense": ("chi phi lai vay", "interest expense"),
    "selling_expense": ("chi phi ban hang", "selling expense"),
    "admin_expense": ("chi phi quan ly doanh nghiep", "administrative expense"),
    "profit_before_tax": (
        "loi nhuan truoc thue", "tong loi nhuan ke toan truoc thue",
        "profit before tax", "pbt",
    ),
    "profit_after_tax": (
        "loi nhuan sau thue", "loi nhuan sau thue thu nhap doanh nghiep",
        "lai rong", "loi nhuan rong", "profit after tax", "net income", "pat",
    ),
    "eps": ("lai co ban tren co phieu", "lai tren co phieu", "eps"),
    # ── Balance sheet ──
    "total_assets": ("tong tai san", "tong cong tai san", "total assets"),
    "current_assets": ("tai san ngan han", "current assets"),
    "non_current_assets": ("tai san dai han", "non-current assets"),
    "cash": ("tien va cac khoan tuong duong tien", "tien va tuong duong tien", "cash"),
    "inventory": ("hang ton kho", "inventory", "inventories"),
    "receivables": ("cac khoan phai thu", "phai thu khach hang", "receivables"),
    "total_liabilities": ("no phai tra", "tong no phai tra", "total liabilities"),
    "current_liabilities": ("no ngan han", "current liabilities"),
    "non_current_liabilities": ("no dai han", "non-current liabilities"),
    "equity": (
        "von chu so huu", "tong von chu so huu", "nguon von chu so huu",
        "equity", "shareholders equity",
    ),
    "charter_capital": ("von dieu le", "von gop cua chu so huu", "von dau tu cua chu so huu", "charter capital"),
    "retained_earnings": ("loi nhuan sau thue chua phan phoi", "retained earnings"),
    # ── Cash flow ──
    "cf_operating": (
        "luu chuyen tien thuan tu hoat dong kinh doanh",
        "tien thuan tu hoat dong kinh doanh", "operating cash flow",
    ),
    "cf_investing": (
        "luu chuyen tien thuan tu hoat dong dau tu", "investing cash flow",
    ),
    "cf_financing": (
        "luu chuyen tien thuan tu hoat dong tai chinh", "financing cash flow",
    ),
    "depreciation": ("khau hao", "khau hao tai san co dinh", "depreciation"),
    # ── Ngan hang (Banking) ──
    "net_interest_income": (
        "thu nhap lai thuan", "thu nhap lai va cac khoan thu nhap tuong tu thuan", "lai thuan", "net interest income",
    ),
    "customer_loans": (
        "cho vay khach hang", "du no cho vay", "du no cho vay khach hang", "loans to customers", "so du cho vay khach hang",
    ),
    "customer_deposits": ("tien gui cua khach hang", "tien gui khach hang", "deposits from customers"),
    "provision_expense": (
        "chi phi du phong", "chi phi du phong rui ro tin dung", "du phong rui ro", "credit provision expense",
    ),
    # ── Chung khoan & Dau tu (Securities) ──
    "fvtpl": ("fvtpl", "tai san tai chinh ghi nhan thong qua lai lo", "lai tu fvtpl", "lo tu fvtpl"),
    "htm": ("htm", "dau tu nam giu den ngay dao han", "held to maturity"),
    "afs": ("afs", "tai san tai chinh san sang de ban", "available for sale"),
    "brokerage_revenue": ("doanh thu nghiep vu moi gioi", "doanh thu moi gioi chung khoan", "doanh thu moi gioi"),
    # ── Bat dong san & Xay dung (Real Estate) ──
    "unearned_revenue": (
        "nguoi mua tra tien truoc", "nguoi mua tra tien truoc ngan han", "doanh thu chua thuc hien", "unearned revenue",
    ),
    "construction_in_progress": ("chi phi xay dung co ban do dang", "xay dung do dang", "construction in progress"),
    "investment_properties": ("bat dong san dau tu", "investment properties"),
    # ── Thuyet minh chi tiet (Notes & Details) ──
    "deposit_interest_income": ("lai tien gui", "lai tien gui ngan hang", "thu lai tien gui", "interest on deposits"),
    "penalty_expense": ("chi phi phat", "tien phat", "tien phat vi pham", "chi phi phat thue", "tien phat cham nop"),
    "staff_expense": ("chi phi nhan vien", "chi phi luong", "luong va cac khoan theo luong", "staff costs"),
    "tangible_fixed_assets": ("tai san co dinh huu hinh", "tscd huu hinh", "tangible fixed assets"),
    "fixed_assets_cost": ("nguyen gia", "nguyen gia tai san co dinh", "historical cost"),
    "accumulated_depreciation": ("gia tri hao mon luy ke", "hao mon luy ke", "accumulated depreciation"),
}

# Chi so dan xuat — khong co san trong bang, phai TINH
DERIVED_METRICS: dict[str, tuple[str, ...]] = {
    "roe": ("roe", "ty suat loi nhuan tren von chu so huu", "loi nhuan tren von chu so huu"),
    "roa": ("roa", "ty suat loi nhuan tren tong tai san", "loi nhuan tren tong tai san"),
    "ros": ("ros", "ty suat loi nhuan tren doanh thu"),
    "debt_to_equity": (
        "ty le no tren von chu so huu", "no tren von chu so huu", "d/e", "debt to equity",
    ),
    "current_ratio": ("ty so thanh toan hien hanh", "he so thanh toan ngan han", "current ratio"),
    "quick_ratio": ("ty so thanh toan nhanh", "quick ratio"),
    "gross_margin": ("ty le loi nhuan gop", "bien loi nhuan gop", "gross margin"),
    "net_margin": ("ty le loi nhuan rong", "bien loi nhuan rong", "net margin"),
    "growth": ("tang truong", "toc do tang truong", "tang truong so voi", "growth rate"),
    "cagr": ("cagr", "tang truong kep"),
}

# Cong thuc tinh — nhung vao prompt sinh pandas de LLM khong tu bia
DERIVED_FORMULAS: dict[str, str] = {
    "roe": "profit_after_tax / equity * 100.0",
    "roa": "profit_after_tax / total_assets * 100.0",
    "ros": "profit_after_tax / net_revenue * 100.0",
    "debt_to_equity": "total_liabilities / equity",
    "current_ratio": "current_assets / current_liabilities",
    "quick_ratio": "(current_assets - inventory) / current_liabilities",
    "gross_margin": "gross_profit / net_revenue * 100.0",
    "net_margin": "profit_after_tax / net_revenue * 100.0",
    "growth": "(value_year_n - value_year_n_minus_1) / abs(value_year_n_minus_1) * 100.0",
    "cagr": "((value_end / value_start) ** (1 / n_years) - 1) * 100.0",
}

# Cac thanh phan nguyen tu cua chi tieu phai sinh de mo rong Schema Linking
DERIVED_SUB_TERMS: dict[str, list[str]] = {
    "roe": ["lợi nhuận sau thuế", "vốn chủ sở hữu"],
    "roa": ["lợi nhuận sau thuế", "tổng tài sản"],
    "ros": ["lợi nhuận sau thuế", "doanh thu thuần"],
    "gross_margin": ["lợi nhuận gộp", "doanh thu thuần"],
    "net_margin": ["lợi nhuận sau thuế", "doanh thu thuần"],
    "debt_to_equity": ["nợ phải trả", "vốn chủ sở hữu"],
    "current_ratio": ["tài sản ngắn hạn", "nợ ngắn hạn"],
    "quick_ratio": ["tài sản ngắn hạn", "hàng tồn kho", "nợ ngắn hạn"],
}


@dataclass(slots=True)
class TermMatch:
    metric: str
    surface: str          # bien the da khop
    is_derived: bool
    position: int         # vi tri trong text — de sap xep theo thu tu xuat hien


class TermMapper:
    """Khop cum tu tai chinh trong text tu do.

    Uu tien cum DAI nhat de "doanh thu thuan" khong bi "doanh thu" chiem cho.
    """

    def __init__(self) -> None:
        self._index: list[tuple[str, str, bool]] = []  # (surface, metric, is_derived)
        for metric, surfaces in FINANCIAL_TERMS.items():
            for s in surfaces:
                self._index.append((normalize_text(s), metric, False))
        for metric, surfaces in DERIVED_METRICS.items():
            for s in surfaces:
                self._index.append((normalize_text(s), metric, True))
        # Dai truoc, ngan sau
        self._index.sort(key=lambda x: len(x[0]), reverse=True)

    def find_all(self, text: str) -> list[TermMatch]:
        """Tim moi metric xuat hien, khong trung lap, uu tien cum dai."""
        flat = normalize_text(text)
        taken: list[tuple[int, int]] = []
        out: list[TermMatch] = []

        for surface, metric, derived in self._index:
            start = flat.find(surface)
            if start < 0:
                continue
            end = start + len(surface)
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            out.append(TermMatch(metric, surface, derived, start))

        out.sort(key=lambda m: m.position)
        return out

    def map_one(self, text: str) -> str | None:
        matches = self.find_all(text)
        return matches[0].metric if matches else None

    def metrics(self, text: str) -> list[str]:
        seen: dict[str, None] = {}
        for m in self.find_all(text):
            seen.setdefault(m.metric, None)
        return list(seen)

    def has_derived(self, text: str) -> bool:
        return any(m.is_derived for m in self.find_all(text))

    @staticmethod
    def formula(metric: str) -> str | None:
        return DERIVED_FORMULAS.get(metric)
