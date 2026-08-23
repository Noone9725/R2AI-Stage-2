"""Anh xa TEN CONG TY tieng Viet -> ma chung khoan (ticker).

TAI SAO CAN: do tren bo cau hoi that (1012 cau), 362 cau KHONG viet ma CK
ma goi ten day du — "Ngan hang TMCP A Chau" thay vi "ACB". Regex bat ma
viet hoa truot het cac cau nay, nen hard filter theo ticker mat 36% cau
hoi: khong loc duoc thi phai xet toan bo ~157k bang.

BTC cung cap san bang anh xa tai `data/questions/code_stock.csv`
(cot "Ma CK", "Ten cong ty") — dung chinh no lam nguon su that, khong
doan ten.

Cach so khop: bo dau + ha chu thuong (OCR/cau hoi khac dau nhau), roi
tim ten cong ty nhu MOT CHUOI CON cua cau hoi. Ten dai nhat thang, vi
"Ngan hang TMCP Sai Gon - Ha Noi" (SHB) va "Ngan hang TMCP Sai Gon Cong
Thuong" (SGB) dung chung tien to "ngan hang tmcp sai gon".

Ngoai ten phap ly day du, cau hoi con goi ten thuong goi ("Hoa Phat",
"Vinamilk"). Hai nguon alias bu vao: hau to cua ten phap ly khi hau to
do DUY NHAT tren ca 100 cong ty, va `_BRAND_ALIASES` cho ten thuong mai
khong suy ra duoc tu ten phap ly. Ket qua tren 1012 cau hoi that: 1011
cau loc duoc ticker, khong sinh ma gia nao.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..utils.logging import get_logger
from ..utils.vn_text import normalize_text

log = get_logger(__name__)

# Tu chi loai hinh phap ly — khong phan biet cong ty nao voi cong ty nao.
# Bo chung di de "CTCP Tap doan Hoa Phat" vs "Cong ty Co phan Tap doan
# Hoa Phat" (hai cach viet cua cung mot doanh nghiep) cung khop.
_LEGAL_NOISE = (
    "cong ty co phan", "cong ty tnhh mtv", "cong ty tnhh", "tong cong ty",
    "ctcp", "tmcp", "mtv", "tnhh", "cong ty",
)
_PUNCT_RE = re.compile(r"[^0-9a-z]+")

# Ten giao dich KHONG suy ra duoc tu ten phap ly trong code_stock.csv.
# Do tren bo cau hoi that: cac cach goi nay xuat hien thay cho ma CK.
_BRAND_ALIASES: dict[str, str] = {
    "vinamilk": "VNM",           # CTCP Sua Viet Nam
    "eximbank": "EIB",           # NH TMCP Xuat nhap khau Viet Nam
    "mbbank": "MBB",             # NH TMCP Quan doi
    "mb bank": "MBB",
    "saigonbank": "SGB",         # NH TMCP Sai Gon Cong Thuong
    "vietcombank": "VCB",
    "vietinbank": "CTG",
    "techcombank": "TCB",
    "sacombank": "STB",
    "agribank": "AGR",
    "bidv": "BID",
    "vpbank": "VPB",
    "acb": "ACB",
    "vingroup": "VIC",
    # Ten thuong goi MOT tu — alias suy dien bi cam o do dai nay (xem
    # _MIN_ALIAS_TOKENS), nen phai liet ke tay.
    "masan": "MSN",
    "gelex": "GEX",
}

# Tu qua chung, khong duoc lam alias mot minh du chi xuat hien o mot ten.
_GENERIC_TOKENS = frozenset({
    "xanh", "viet", "nam", "vietnam", "dau", "khi", "dien", "luc", "thep",
    "sua", "nhua", "dam", "than", "cao", "su", "xay", "dung", "dich", "vu",
    "dau tu", "phat trien", "tap doan", "san xuat", "thuong mai",
    "viet nam", "ha noi", "sai gon", "ho chi minh", "thanh pho ho chi minh",
    "mien nam", "mien bac", "mien trung", "ngan hang", "chung khoan",
    "bat dong san", "cong nghiep", "cong nghe", "quoc te", "xuat nhap khau",
    # SNZ = "Tong CTCP Phat trien Khu Cong nghiep" — ten phap ly hoan toan
    # la tu chung. Khong duoc sinh alias tu no, neu khong moi cau hoi noi
    # ve "khu cong nghiep" deu bi gan SNZ. Liet ke ca tung tu de phep
    # kiem "moi tu deu chung" bat duoc MOI hau to cua ten nay.
    "khu", "phat", "trien", "cong", "nghiep", "tong", "quan", "ly",
    "khu cong nghiep", "phat trien khu cong nghiep",
})
# Alias suy ra tu duoi ten phap ly chi lay toi 4 tu — dai hon thi da la
# chuoi con cua chinh ten day du, khong them duoc gi.
#
# San duoi la 2 TU: hau to mot tu ("tien" trong "Xi Mang Vicem Ha Tien",
# "nhuan" trong "Vang bac Da quy Phu Nhuan") la tu tieng Viet thong thuong
# va khop bua vao hang tram cau hoi khong lien quan. Do tren bo cau hoi
# that: cho phep hau to mot tu lam HT1 khop 203 cau, PNJ 179 cau.
# Ten thuong goi mot tu thuc su (Masan, GELEX) nam trong _BRAND_ALIASES.
_MAX_ALIAS_TOKENS = 4
_MIN_ALIAS_TOKENS = 2
_MIN_ALIAS_CHARS = 6


def _key(text: str) -> str:
    """Chuan hoa de so khop chuoi con: bo dau, bo tu phap ly, gom space."""
    flat = normalize_text(text)
    for noise in _LEGAL_NOISE:
        flat = flat.replace(noise, " ")
    return _PUNCT_RE.sub(" ", flat).strip()


DEFAULT_VN_TICKERS = frozenset({
    "AAA", "AAM", "AAS", "ABB", "ABC", "ACB", "ACC", "ACL", "AGG", "AGR",
    "AMD", "ANV", "APC", "APG", "APH", "ASM", "ASP", "AST", "BCE", "BCG",
    "BFC", "BIC", "BID", "BKG", "BMC", "BMI", "BMP", "BSI", "BSR", "BTP",
    "BTT", "BVH", "BWE", "C32", "C47", "CCL", "CDC", "CHP", "CII", "CLW",
    "CMX", "CNG", "COM", "CRE", "CSM", "CSV", "CTD", "CTF", "CTG", "CTR",
    "CTS", "D2D", "DAG", "DAH", "DBC", "DBD", "DBW", "DCM", "DGC", "DGW",
    "DHA", "DHC", "DHG", "DIG", "DLG", "DLR", "DMC", "DNP", "DPC", "DPG",
    "DPM", "DPR", "DRC", "DRL", "DS3", "DSN", "DST", "DTA", "DTL", "DTP",
    "DTT", "DVP", "DXG", "DXS", "EIB", "ELC", "EMC", "EVE", "EVF", "EVG",
    "FCN", "FDC", "FIR", "FIT", "FLC", "FMC", "FPT", "FRT", "FTS", "GAS",
    "GDT", "GEG", "GEX", "GIL", "GMC", "GMD", "GSP", "GTA", "GVR", "HAH",
    "HAI", "HAP", "HAR", "HAX", "HBC", "HCD", "HCM", "HDB", "HDC", "HDG",
    "HHP", "HHS", "HII", "HMC", "HNG", "HPX", "HPG", "HSG", "HT1", "HTI",
    "HTL", "HTN", "HU1", "HUB", "HU4", "HUT", "HVG", "HVH", "HVN", "HVX",
    "IBC", "ICT", "IDI", "IJC", "IMP", "ITA", "ITC", "ITD", "JVC", "KBC",
    "KDC", "KDH", "KHP", "KMR", "KOS", "KPF", "KSB", "L10", "L14", "L18",
    "LAF", "LBM", "LCG", "LDG", "LEC", "LGC", "LGL", "LHG", "LIX", "LMH",
    "LPB", "MBB", "MCG", "MCP", "MDG", "MHC", "MIG", "MSB", "MSN", "MSR",
    "MWG", "NAF", "NAV", "NBC", "NBB", "NCT", "NHA", "NHH", "NHT", "NKG",
    "NLG", "NNC", "NO1", "NSC", "NT2", "NTL", "NVL", "OCB", "OGC", "OPC",
    "PAC", "PAN", "PC1", "PCG", "PDN", "PDR", "PET", "PGC", "PGD", "PGI",
    "PGV", "PHC", "PHR", "PIT", "PJC", "PJT", "PLC", "PLX", "PNJ", "POM",
    "POW", "PPC", "PRT", "PSB", "PSD", "PSI", "PSW", "PTC", "PTL", "PVD",
    "PVI", "PVO", "PVP", "PVT", "PXI", "PXT", "QCG", "RAL", "RDP", "REE",
    "RIC", "S4A", "SAB", "SAM", "SAV", "SBA", "SBG", "SBL", "SBT", "SBV",
    "SC5", "SCD", "SCR", "SCS", "SFC", "SFG", "SFI", "SGN", "SGR", "SGS",
    "SHA", "SHB", "SHI", "SHP", "SHS", "SJF", "SJS", "SKG", "SMA", "SMB",
    "SMC", "SPM", "SRC", "SRF", "SSB", "SSI", "ST8", "STB", "STC", "STG",
    "STK", "SVC", "SVI", "SVT", "SZC", "SZL", "TAC", "TAR", "TBX", "TCB",
    "TCD", "TCH", "TCM", "TCO", "TCS", "TDG", "TDH", "TDM", "TDP", "TDW",
    "TEG", "TGG", "THG", "TIP", "TIX", "TLG", "TLH", "TN1", "TNA", "TNC",
    "TNH", "TNI", "TNT", "TPB", "TPC", "TPG", "TRA", "TRC", "TSC", "TTA",
    "TTB", "TTF", "TV2", "TV3", "TV4", "TVB", "TVC", "TVS", "TYA", "UDJ",
    "UIC", "VAF", "VCA", "VCB", "VCG", "VCI", "VDP", "VDS", "VFG", "VGC",
    "VGG", "VGI", "VGL", "VGS", "VGT", "VHC", "VHM", "VIB", "VIC", "VID",
    "VIP", "VIS", "VIX", "VJC", "VKC", "VMD", "VNE", "VNG", "VNL", "VNM",
    "VNP", "VNS", "VOS", "VPB", "VPD", "VPG", "VPH", "VPI", "VPS", "VRC",
    "VRE", "VSC", "VSH", "VSI", "VTB", "VTC", "VTD", "VTO", "VTP", "VTR",
    "VWS", "VXB", "YEG"
})


class CompanyMap:
    """Tra ticker tu ten cong ty xuat hien trong cau hoi."""

    def __init__(self, csv_path: str | Path | None = None):
        paths_to_try = []
        if csv_path:
            paths_to_try.append(Path(csv_path))
        else:
            settings = get_settings()
            paths_to_try.extend([
                settings.paths.questions / "code_stock.csv",
                settings.root / "data" / "questions" / "code_stock.csv",
                settings.root / "data" / "code_stock.csv",
                settings.paths.raw / "questions" / "code_stock.csv",
                settings.paths.raw / "code_stock.csv",
            ])

        found_path: Path | None = None
        for p in paths_to_try:
            if p.exists():
                found_path = p
                break

        self._entries: list[tuple[str, str]] = []  # (key ten cong ty, ticker)
        self.tickers: set[str] = set()

        if found_path and found_path.exists():
            with found_path.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    vals = [v.strip() for v in row.values()]
                    if len(vals) < 2 or not vals[0]:
                        continue
                    ticker, name = vals[0].upper(), vals[1]
                    self.tickers.add(ticker)
                    key = _key(name)
                    if key:
                        self._entries.append((key, ticker))
        elif not csv_path:
            # Chi fallback danh sach default khi chay pipeline tong the va khong truyen file rieng
            self.tickers = set(DEFAULT_VN_TICKERS)

        # Ten dai truoc: tien to chung khong duoc thang ten cu the hon.
        self._entries.sort(key=lambda kv: len(kv[0]), reverse=True)
        self._add_aliases()
        log.debug("CompanyMap: %d ten -> %d ticker", len(self._entries), len(self.tickers))

    def _add_aliases(self) -> None:
        """Them ten goi tat: hau to ten phap ly + bang brand viet tay.

        "Trong nhom Hoa Phat, Hoa Sen va Nam Kim" khong chua ten phap ly
        nao ("CTCP Tap doan Hoa Phat"), nhung DUOI ten phap ly chinh la
        ten thuong goi. Lay cac hau to 2..4 tu lam alias ung vien (hau to
        mot tu bi cam — xem _MIN_ALIAS_TOKENS).

        Chi giu ung vien DUY NHAT tren ca 100 cong ty. "tap doan" xuat
        hien o hang chuc ten, "hoa phat" chi o mot — cai dau bi loai, cai
        sau thanh alias. Nho vay khong can doan cong ty nao "quan trong".
        """
        seen = {key for key, _ in self._entries}
        counts: dict[str, set[str]] = {}
        for key, ticker in list(self._entries):
            toks = key.split()
            for n in range(_MIN_ALIAS_TOKENS, min(_MAX_ALIAS_TOKENS, len(toks)) + 1):
                cand = " ".join(toks[-n:])
                if cand in seen or len(cand) < _MIN_ALIAS_CHARS:
                    continue
                if cand in _GENERIC_TOKENS or all(t in _GENERIC_TOKENS for t in cand.split()):
                    continue
                counts.setdefault(cand, set()).add(ticker)

        extra = [(c, next(iter(t))) for c, t in counts.items() if len(t) == 1]

        # Brand viet tay: khong suy ra duoc tu ten phap ly, va duoc uu tien
        # hon alias suy dien neu trung (vd "acb" da la ma CK).
        for brand, ticker in _BRAND_ALIASES.items():
            if ticker in self.tickers:
                extra.append((_key(brand), ticker))

        self._entries.extend(extra)
        self._entries.sort(key=lambda kv: len(kv[0]), reverse=True)

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, text: str) -> list[str]:
        """Moi ticker co ten cong ty xuat hien trong `text`, thu tu on dinh.

        Khi mot doan da khop mot ten, doan do bi che lai de ten ngan hon
        long trong no khong khop them lan nua.
        """
        hay = f" {_key(text)} "
        out: list[str] = []
        for key, ticker in self._entries:
            needle = f" {key} "
            pos = hay.find(needle)
            if pos == -1 or ticker in out:
                continue
            out.append(ticker)
            hay = hay[: pos + 1] + " " * len(key) + hay[pos + 1 + len(key) :]
        return out


@lru_cache(maxsize=1)
def get_company_map() -> CompanyMap:
    """Ban dung chung — doc CSV mot lan cho ca 1012 cau hoi."""
    return CompanyMap()
