"""Don vi cua DAP AN — suy tu cau hoi, ap o cuoi pipeline.

TAI SAO CAN (do tren 1012 cau hoi that):
    391 cau hoi "ty dong", 213 cau "trieu dong", 16 cau "nghin dong"
    -> 620/1012 = 61.3% cau hoi can chia lai.

Pipeline chuan hoa MOI gia tri ve VND o `schema_std.py` (input-side, giu
nguyen — do la representation noi bo dung). Nhung cau hoi lai hoi bang
don vi khac. Khong co buoc chuyen nguoc thi dap an lech 1e3..1e9 lan du
retrieval va pandas deu dung.

THIET KE: deterministic, o BOUNDARY, khong nho LLM.
    raw_result (VND) -> AnswerNormalizer.to_asked_unit() -> final answer

KHONG nhung he so chia vao code pandas sinh ra: LLM hay quen, hay chia
hai lan, va khi sua loi (self-repair) thi he so bien mat. Chia mot lan o
`SubmissionBuilder` la diem duy nhat khong the truot.

PHAN BIET `%` VA `lan` — hai semantic KHAC NHAU:
    "ty suat loi nhuan ... bao nhieu %"   -> ratio 0.153 -> 15.3
    "he so thanh toan ... bao nhieu lan"  -> ratio 1.53  -> 1.53  (GIU NGUYEN)
Gop chung se lam sai ca 115 cau `%` lan 51 cau `lan`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..utils.vn_text import normalize_text


class AskedUnit(str, Enum):
    """Don vi ma cau hoi yeu cau o dap an."""

    DONG = "dong"                    # VND — khong doi
    NGHIN_DONG = "nghin_dong"        # /1e3
    TRIEU_DONG = "trieu_dong"        # /1e6
    TY_DONG = "ty_dong"              # /1e9
    TRAM_TY_DONG = "tram_ty_dong"    # /1e11
    NGHIN_TY_DONG = "nghin_ty_dong"  # /1e12
    PERCENT = "percent"              # ratio -> phan tram
    TIMES = "times"                  # "lan" — ty so, GIU NGUYEN
    NONE = "none"                    # cau hoi khong neu don vi


# Chia cho he so nay de doi tu VND sang don vi duoc hoi.
# PERCENT/TIMES khong phai don vi tien te -> khong nam o day.
MONETARY_DIVISORS: dict[AskedUnit, float] = {
    AskedUnit.DONG: 1.0,
    AskedUnit.NGHIN_DONG: 1e3,
    AskedUnit.TRIEU_DONG: 1e6,
    AskedUnit.TY_DONG: 1e9,
    AskedUnit.TRAM_TY_DONG: 1e11,
    AskedUnit.NGHIN_TY_DONG: 1e12,
}

# THU TU QUAN TRONG.
# 1. TIMES: "bao nhieu lan", "he so ... lan"
# 2. PERCENT: "diem phan tram", "phan tram", "%" (uu tien truoc dong/tien te de chong dinh chu "dong" trong "dong tien")
# 3. TIEN TE: nghin ty dong -> tram ty dong -> ty dong -> trieu dong -> nghin dong -> dong
_UNIT_PATTERNS: tuple[tuple[re.Pattern[str], AskedUnit], ...] = (
    (re.compile(r"\bbao nhieu lan\b"), AskedUnit.TIMES),
    (re.compile(r"\b(?:la|bang|dat)\s+bao nhieu\s+lan\b"), AskedUnit.TIMES),
    (re.compile(r"\bhe so\b.*?\blan\b"), AskedUnit.TIMES),
    (re.compile(r"\b(?:diem\s+)?phan\s+tram\b"), AskedUnit.PERCENT),
    (re.compile(r"\bbao nhieu\s*%"), AskedUnit.PERCENT),
    (re.compile(r"\bty le\s+.*?(?:phan\s+tram|%)"), AskedUnit.PERCENT),
    (re.compile(r"%"), AskedUnit.PERCENT),
    (re.compile(r"\b(?:nghin|ngan)\s+ty\s*dong\b"), AskedUnit.NGHIN_TY_DONG),
    (re.compile(r"\b(?:nghin|ngan)\s+ty\b"), AskedUnit.NGHIN_TY_DONG),
    (re.compile(r"\btram\s+ty\s*dong\b"), AskedUnit.TRAM_TY_DONG),
    (re.compile(r"\btram\s+ty\b"), AskedUnit.TRAM_TY_DONG),
    (re.compile(r"\bty\s*dong\b"), AskedUnit.TY_DONG),
    (re.compile(r"\bbao nhieu\s+ty\b"), AskedUnit.TY_DONG),
    (re.compile(r"\btrieu\s*dong\b"), AskedUnit.TRIEU_DONG),
    (re.compile(r"\bbao nhieu\s+trieu\b"), AskedUnit.TRIEU_DONG),
    (re.compile(r"\b(?:nghin|ngan)\s*dong\b"), AskedUnit.NGHIN_DONG),
    (re.compile(r"\bbao nhieu\s+(?:nghin|ngan)\b"), AskedUnit.NGHIN_DONG),
    (re.compile(r"\bbao nhieu\s+dong\b"), AskedUnit.DONG),
    (re.compile(r"\b(?:tinh bang|bang|don vi)\s+dong\b"), AskedUnit.DONG),
    (re.compile(r"\b(?:vnd|viet nam dong)\b"), AskedUnit.DONG),
    (re.compile(r"\blan\b"), AskedUnit.TIMES),
)


def detect_asked_unit(question: str) -> AskedUnit:
    """Don vi cau hoi yeu cau. Khong chac -> NONE (khong doi gi ca)."""
    flat = normalize_text(question)
    for pattern, unit in _UNIT_PATTERNS:
        if pattern.search(flat):
            return unit
    return AskedUnit.NONE


@dataclass(frozen=True)
class NormalizedAnswer:
    """Ket qua sau khi doi don vi — giu ca gia tri goc de debug/log."""

    value: float
    raw_value: float
    unit: AskedUnit
    divisor: float = 1.0
    converted: bool = False


class AnswerNormalizer:
    """Doi ket qua thuc thi (VND / ratio) sang don vi cau hoi yeu cau.

    Dat o BOUNDARY cuoi cung (SubmissionBuilder), sau execution, truoc khi
    lam tron. Deterministic hoan toan — khong goi LLM.
    """

    def __init__(self, round_to: int = 2):
        self.round_to = round_to

    def normalize(
        self, value: float, question: str, *, unit: AskedUnit | None = None
    ) -> NormalizedAnswer:
        asked = unit if unit is not None else detect_asked_unit(question)
        return self.apply(value, asked)

    def apply(
        self, value: float, asked: AskedUnit, *, source_unit: str | None = None, is_fin_inst: bool = False
    ) -> NormalizedAnswer:
        raw = float(value)
        if raw == 0.0:
            return NormalizedAnswer(
                value=0.0, raw_value=0.0,
                unit=asked, divisor=1.0, converted=False,
            )

        abs_raw = abs(raw)

        # 1. Neu bang nguon co don vi ro rang tu manifest (trieu_dong, ty_dong)
        if source_unit in ("trieu_dong", "triệu đồng", "trieu"):
            if asked is AskedUnit.TY_DONG:
                div = 1e3
            elif asked is AskedUnit.TRIEU_DONG:
                div = 1.0
            elif asked is AskedUnit.NGHIN_TY_DONG:
                div = 1e6
            elif asked is AskedUnit.DONG:
                div = 1e-6
            elif asked is AskedUnit.NGHIN_DONG:
                div = 1e-3
            else:
                div = 1.0
            out = raw / div
            val = round(out, self.round_to)
            if val == 0.0 and out != 0.0:
                val = round(out, 4)
            return NormalizedAnswer(value=val, raw_value=raw, unit=asked, divisor=div, converted=div != 1.0)

        if source_unit in ("ty_dong", "tỷ đồng", "ty"):
            if asked is AskedUnit.TY_DONG:
                div = 1.0
            elif asked is AskedUnit.TRIEU_DONG:
                div = 1e-3
            elif asked is AskedUnit.NGHIN_TY_DONG:
                div = 1e3
            else:
                div = 1.0
            out = raw / div
            val = round(out, self.round_to)
            if val == 0.0 and out != 0.0:
                val = round(out, 4)
            return NormalizedAnswer(value=val, raw_value=raw, unit=asked, divisor=div, converted=div != 1.0)

        # 2. Neu table_unit la null hoac 'vnd': Danh gia dua tren Magnitude logic
        div = 1.0

        if asked is AskedUnit.TY_DONG:
            # So >= 1e8 (>= 100 trieu VND) -> chac chan la VND -> chia 1e9
            if abs_raw >= 1e8:
                div = 1e9
            # So 10.0 <= raw < 1e8 (vi du FTS 444918 tr) -> ban chat da la trieu dong -> chia 1e3
            elif abs_raw >= 10.0:
                div = 1e3
            else:
                div = 1.0
            out = raw / div
            val = round(out, self.round_to)
            if val == 0.0 and out != 0.0:
                val = round(out, 4)
            return NormalizedAnswer(value=val, raw_value=raw, unit=asked, divisor=div, converted=div != 1.0)

        elif asked is AskedUnit.TRIEU_DONG:
            # Cac ngan hang / cong ty chung khoan thuong co bang lap san don vi trieu dong
            if is_fin_inst and abs_raw < 1e9:
                div = 1.0
            elif abs_raw >= 1e6:
                div = 1e6
            else:
                div = 1.0
            out = raw / div
            val = round(out, self.round_to)
            if val == 0.0 and out != 0.0:
                val = round(out, 4)
            return NormalizedAnswer(value=val, raw_value=raw, unit=asked, divisor=div, converted=div != 1.0)

        elif asked is AskedUnit.NGHIN_TY_DONG:
            if abs_raw >= 1e11:
                div = 1e12
            elif abs_raw >= 1e6:
                div = 1e3
            else:
                div = 1.0
            out = raw / div
            val = round(out, self.round_to)
            return NormalizedAnswer(value=val, raw_value=raw, unit=asked, divisor=div, converted=div != 1.0)

        elif asked is AskedUnit.NGHIN_DONG:
            div = 1e3 if abs_raw >= 1e3 else 1.0
            out = raw / div
            val = round(out, self.round_to)
            return NormalizedAnswer(value=val, raw_value=raw, unit=asked, divisor=div, converted=div != 1.0)

        elif asked is AskedUnit.PERCENT:
            val = round(raw, self.round_to)
            if val == 0.0 and raw != 0.0:
                val = round(raw, 4)
            return NormalizedAnswer(
                value=val, raw_value=raw,
                unit=asked, divisor=1.0, converted=False,
            )

        # TIMES va NONE: ty so / khong ro don vi -> giu nguyen.
        val = round(raw, self.round_to)
        if val == 0.0 and raw != 0.0:
            val = round(raw, 4)
        return NormalizedAnswer(
            value=val, raw_value=raw,
            unit=asked, divisor=1.0, converted=False,
        )


def divisor_for(question: str) -> float:
    """He so chia ung voi cau hoi — tien cho prompt/log."""
    return MONETARY_DIVISORS.get(detect_asked_unit(question), 1.0)
