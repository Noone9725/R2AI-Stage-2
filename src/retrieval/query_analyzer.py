"""Trich ticker / nam / chi tieu tu cau hoi tieng Viet.

Ket qua dung cho hard filter — buoc cat giam khong gian tim kiem manh nhat.
Sai o day la sai ca chuoi, nen moi rule deu uu tien PHU HON la chac:
khong chac ticker thi tra ve rong (khong loc) con hon loc nham.
"""

from __future__ import annotations

import re

from ..normalization.answer_unit import detect_asked_unit
from ..normalization.period import detect_requested_period
from ..normalization.term_mapper import TermMapper
from ..schemas import Question
from ..utils.vn_text import normalize_text
from .company_map import CompanyMap, get_company_map

# Ticker trong ngoac: "Công ty CP Sữa Việt Nam (VNM)"
_TICKER_PAREN_RE = re.compile(r"\(([A-Z]{3,4}\d?)\)")
# Ticker dung mot minh giua cau, viet hoa. Cho phep MOT chu so o cuoi:
# HT1, PC1, TV2 la ma that va xuat hien nhieu lan trong bo cau hoi.
_TICKER_BARE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{3,4}\d?)(?![A-Za-z0-9])")
# Ma viet THUONG: "trong ngành BĐS (gồm các công ty hpx,kbc,nvl,vic,vpi,vre)".
# Chi nhan khi khop danh sach ma that — chu tieng Viet 3-4 ky tu ("cua",
# "nam", "tren") se trung vo ke neu nhan bua.
_TICKER_LOWER_RE = re.compile(r"(?<![A-Za-z0-9])([a-z]{3,4}\d?)(?![A-Za-z0-9])")
# Ma co chu so nhung CHI 2 chu cai: HT1, PC1, TV2. `_TICKER_BARE_RE` doi
# toi thieu 3 chu cai nen truot het. Chi nhan khi khop danh sach ma that.
_TICKER_ALNUM_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,4}\d{1,2})(?![A-Za-z0-9])")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_YEAR_RANGE_RE = re.compile(
    r"(?:(?:tu|từ|giai\s*doan|giai\s*đoạn)\s*(?:nam|năm)?\s*)?((?:19|20)\d{2})"
    r"\s*(?:den|đến|[-–]|toi|tới)\s*(?:nam|năm)?\s*((?:19|20)\d{2})",
    re.IGNORECASE,
)

# Viet tat hay bi nham la ticker
_FALSE_TICKERS = frozenset({
    "CP", "CTCP", "TNHH", "MTV", "BCTC", "VND", "USD", "EUR", "ROE", "ROA",
    "ROS", "EPS", "EBIT", "CAGR", "AI", "GDP", "VAT", "TSCD", "LNST", "DTT",
    "TMCP", "NHNN", "HDQT", "HN", "CTY", "TSDH", "TSNH", "VCSH", "EBITDA",
    "CFO", "TNDN", "LNTT", "LDR", "GTCG", "BOT",
})

_LOWER_WORD_BLOCKLIST = frozenset({
    "hai", "nam", "sau", "ban", "cho", "vay", "thu", "chi", "moi", "tam", "chin", "mot", "tram", "ngan", "tong",
    "bao", "nhieu", "trieu", "dong", "khach", "hang", "trong", "theo", "duoc", "nhom", "tren", "duoi"
})

_COMPARE_HINTS = ("so voi", "so sanh", "cao hon", "thap hon", "chenh lech", "hon kem")
_AGGREGATE_HINTS = ("tong", "trung binh", "binh quan", "cao nhat", "thap nhat", "lon nhat")


class QueryAnalyzer:
    def __init__(
        self,
        term_mapper: TermMapper | None = None,
        company_map: CompanyMap | None = None,
    ):
        self.terms = term_mapper or TermMapper()
        self.companies = company_map or get_company_map()

    def analyze(self, question_id: int, text: str) -> Question:
        return Question(
            id=question_id,
            question=text,
            tickers=self.extract_tickers(text),
            years=self.extract_years(text),
            metrics=self.terms.metrics(text),
            needs_derived=self.terms.has_derived(text),
            asked_unit=detect_asked_unit(text).value,
            requested_period=(p.value if (p := detect_requested_period(text)) else ""),
            report_type=self.extract_report_type(text),
        )

    def extract_report_type(self, text: str) -> str | None:
        """Nhan dien yeu cau BCTC rieng le (cong ty me) hay hop nhat."""
        flat = normalize_text(text)
        if any(h in flat for h in ("cong ty me", "rieng le", "bao cao rieng", "bctc rieng", "cua me", "cua rieng")):
            return "separate"
        if any(h in flat for h in ("hop nhat", "toan tap doan", "bctc hn", "bctchn")):
            return "consolidated"
        return None

    # ── ticker ────────────────────────────────────────────

    def extract_tickers(self, text: str) -> list[str]:
        """Ma CK trong cau hoi, uu tien tin hieu chac chan nhat truoc.

        Bon nguon, do dan tren 1012 cau hoi that: regex ma viet hoa chi bat
        duoc 650 cau; them ten cong ty (`code_stock.csv`) len 992; them ma
        co chu so (HT1/PC1) va ten giao dich ngan (Vinamilk, Hoa Phat)
        dong nap 1012 — moi cau hoi loc duoc it nhat mot ma.

        MOI ung vien deu phai co trong `code_stock.csv`. Kho tai lieu chua
        dung 100 thu muc ticker trung khop CSV do, nen ma la ngoai danh
        sach khong bao gio khop bang nao — nhan no chi lam ban ket qua.
        Loc nay chan cac viet tat tai chinh viet hoa dung dang ma CK:
        CFO, TNDN, LNTT, LDR, GTCG, BOT... (14 ma gia, 44 luot).
        """
        known = self.companies.tickers
        found: dict[str, None] = {}

        def take(code: str, is_paren: bool = False) -> None:
            code = code.upper()
            if code in _FALSE_TICKERS:
                return
            if known and code not in known and not is_paren:
                return
            found.setdefault(code, None)

        # 1. Ticker trong ngoac don: "(VJC)" -> rat dang tin cay
        for m in _TICKER_PAREN_RE.finditer(text):
            take(m.group(1), is_paren=True)

        # 2. Ten cong ty day du / ten giao dich ngan: "Ngan hang TMCP A Chau"
        # -> ACB, "Hoa Phat" -> HPG, "CTCP Chung khoan FPT" -> FTS.
        resolved = self.companies.resolve(text)
        for code in resolved:
            found.setdefault(code, None)

        # Mask cac ten cong ty da nhan dien de tranh bare ticker bat nham sub-word
        masked_text = text
        if resolved:
            for key, ticker in self.companies._entries:
                if ticker in resolved and key in masked_text.lower():
                    # Xoa khoang ten da khop khoi chuoi de bare ticker khong bat lai
                    pattern = re.escape(key)
                    masked_text = re.sub(pattern, " " * len(key), masked_text, flags=re.IGNORECASE)

        # 3. Ticker dung doc lap viet hoa: "VJC", "HPG", "ACB"
        if not found:
            for m in _TICKER_BARE_RE.finditer(masked_text):
                take(m.group(1))

        # 4. Ma viet thuong hoac so: chi nhan khi code THUC SU nam trong danh sach ticker va khong phai tu tieng Viet thong thuong
        for m in _TICKER_LOWER_RE.finditer(masked_text):
            word = m.group(1).lower()
            if word in _LOWER_WORD_BLOCKLIST:
                continue
            c = word.upper()
            if known and c in known and c not in _FALSE_TICKERS:
                found.setdefault(c, None)

        # HT1 / PC1 — 2 chu cai + so
        for m in _TICKER_ALNUM_RE.finditer(masked_text):
            c = m.group(1).upper()
            if known and c in known and c not in _FALSE_TICKERS:
                found.setdefault(c, None)

        return list(found)

    # ── year ──────────────────────────────────────────────

    def extract_years(self, text: str) -> list[int]:
        rng = _YEAR_RANGE_RE.search(text)
        if rng:
            lo, hi = int(rng.group(1)), int(rng.group(2))
            if lo > hi:
                lo, hi = hi, lo
            if hi - lo <= 15:
                return list(range(lo, hi + 1))

        years = sorted({int(m.group(1)) for m in _YEAR_RE.finditer(text)})
        return years

    # ── query shape ───────────────────────────────────────

    def is_comparison(self, text: str) -> bool:
        flat = normalize_text(text)
        if any(h in flat for h in _COMPARE_HINTS):
            return True
        return len(self.extract_tickers(text)) > 1 or len(self.extract_years(text)) > 1

    def is_aggregate(self, text: str) -> bool:
        flat = normalize_text(text)
        return any(h in flat for h in _AGGREGATE_HINTS)

    def expand_query(self, question: Question) -> str:
        """Bo sung tin hieu vao query truoc khi search.

        Cau hoi ROE khong chua chu 'loi nhuan sau thue', nhung bang can
        tim thi co. Them thanh phan cong thuc giup BM25/dense bat dung bang.
        """
        parts = [question.question]

        for metric in question.metrics:
            formula = self.terms.formula(metric)
            if formula:
                parts.extend(formula.replace("/", " ").replace("-", " ").split())

        if question.tickers:
            parts.append(" ".join(question.tickers))
        if question.years:
            parts.append(" ".join(str(y) for y in question.years))

        return " ".join(parts)
