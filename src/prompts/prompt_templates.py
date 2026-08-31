"""Load + render prompt template tu configs/prompts/*.txt.

Tach prompt ra file text de sua khong can sua code — vong lap tuning
prompt la vong lap chay nhieu nhat trong cuoc thi nay.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import pandas as pd

from ..config import CONFIG_DIR
from ..normalization.term_mapper import DERIVED_FORMULAS
from ..schemas import RetrievedTable

PROMPT_DIR = CONFIG_DIR / "prompts"

SYSTEM_PANDAS = (
    "Bạn là trợ lý viết code pandas cho dữ liệu báo cáo tài chính Việt Nam. "
    "Bạn luôn trả lời bằng đúng một block code Python, không giải thích."
)

MAX_SAMPLE_ROWS = 6
MAX_ITEM_PREVIEW = 20
MAX_COLUMN_LABELS = 10
MAX_AMBIGUOUS_ITEMS = 4


def _ambiguity_note(df: pd.DataFrame) -> list[str]:
    """Canh bao cac `item` co nhieu dong cung (year, period) — kem cach tach.

    Day la phan quan trong nhat de tranh `.iloc[0]` chon bua. Vi du that
    trong bang can doi: '- Nguyen gia' xuat hien hai lan, la dong con cua
    TSCD huu hinh (ma 222) va TSCD vo hinh (ma 228). Neu chi bao "hay dung
    item_code" thi model khong biet ma nao ton tai; liet ke san candidate
    thi no chi viec chon.

    Chi liet ke toi da vai item — prompt phinh to lam giam chat luong sinh code.
    """
    need = {"item", "year", "period", "value"}
    if not need.issubset(df.columns) or df.empty:
        return []

    grouped = df.groupby(["item", "year", "period"], dropna=False)["value"].nunique()
    ambiguous = grouped[grouped > 1]
    if ambiguous.empty:
        return []

    items: list[str] = []
    for item in dict.fromkeys(str(k[0]) for k in ambiguous.index):
        if len(items) >= MAX_AMBIGUOUS_ITEMS:
            break
        items.append(item)

    lines = [
        "CHÚ Ý — các chỉ tiêu sau có NHIỀU dòng cùng (year, period), "
        "phải lọc thêm để lấy đúng một dòng:"
    ]
    for item in items:
        sub = df[df["item"].astype(str) == item]
        hints: list[str] = []
        codes = [c for c in sub["item_code"].dropna().astype(str).unique() if c.strip()]
        if len(codes) > 1:
            hints.append(f"item_code ∈ {{{', '.join(codes[:6])}}}")
        labels = [
            s for s in sub.get("column_label", pd.Series(dtype=str))
            .dropna().astype(str).unique() if s.strip()
        ]
        if len(labels) > 1:
            hints.append(f"column_label ∈ {{{', '.join(labels[:6])}}}")
        lines.append(
            f"  - `{item}`: " + ("; ".join(hints) if hints else
                                 "các dòng chỉ khác nhau ở thứ tự xuất hiện (row_idx)")
        )
    return lines


@lru_cache(maxsize=16)
def load_template(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Thieu prompt template: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, **kwargs: str) -> str:
    return load_template(name).format(**kwargs)


# ── builder cho tung phan cua prompt ──────────────────────


def format_variables(var_to_csv: dict[str, str]) -> str:
    return "\n".join(f"- {var}: đọc từ {csv}" for var, csv in var_to_csv.items())


def format_schema(var: str, df: pd.DataFrame, question_text: str = "", is_multi_frame: bool = False) -> str:
    """Lieu ke cot + mau dong + danh sach `item` co san."""
    ticker = str(df["ticker"].iloc[0]) if "ticker" in df.columns and not df.empty else ""
    year = str(df["year"].iloc[0]) if "year" in df.columns and not df.empty else ""
    ref = str(df["table_ref"].iloc[0]) if "table_ref" in df.columns and not df.empty else ""
    section = str(df["section"].iloc[0]) if "section" in df.columns and not df.empty and pd.notna(df["section"].iloc[0]) else ""

    sec_name = {
        "income_statement": "Báo cáo Kết quả hoạt động kinh doanh (KQKD)",
        "balance_sheet": "Bảng Cân đối kế toán (CĐKT)",
        "cash_flow": "Báo cáo Lưu chuyển tiền tệ (LCTT)",
        "notes": "Thuyết minh BCTC",
    }.get(section, f"Bảng số liệu {ref}")

    header_info = f"### {var} [{sec_name} | Mã: {ticker} | Năm: {year}]"
    lines = [header_info, f"Số dòng: {len(df)}"]

    if "item" in df.columns:
        items = df["item"].astype(str).drop_duplicates().tolist()
        if question_text:
            stopwords = {"bao", "nhiêu", "tổng", "công", "đồng", "triệu", "nghìn", "ngày", "tháng", "năm", "trong", "theo", "được", "người", "nhóm"}
            q_words = [w.lower() for w in question_text.replace("?", " ").replace(",", " ").split() if len(w) >= 3 and w.lower() not in stopwords]
            def item_score(it: str) -> int:
                it_lower = it.lower()
                return sum(1 for w in q_words if w in it_lower)
            items.sort(key=item_score, reverse=True)

        max_p = 4 if is_multi_frame else MAX_ITEM_PREVIEW
        shown = items[:max_p]
        lines.append("Các giá trị cột `item` liên quan nhất:")
        lines.extend(f"  - {it}" for it in shown)

    if not is_multi_frame:
        if "year" in df.columns:
            years = sorted({str(y) for y in df["year"].dropna().tolist()})
            lines.append(f"Các năm có dữ liệu: {', '.join(years)}")

        if "period" in df.columns:
            periods = [p for p in df["period"].dropna().astype(str).unique().tolist()]
            if periods:
                lines.append(f"Các giá trị cột `period`: {', '.join(sorted(periods))}")

        if "column_label" in df.columns:
            labels = [
                s for s in df["column_label"].dropna().astype(str).unique().tolist()
                if s.strip()
            ]
            if 1 < len(labels) <= MAX_COLUMN_LABELS:
                lines.append(f"Các giá trị cột `column_label`: {', '.join(labels)}")
            elif len(labels) > MAX_COLUMN_LABELS:
                lines.append(
                    f"Cột `column_label` có {len(labels)} giá trị, ví dụ: "
                    f"{', '.join(labels[:MAX_COLUMN_LABELS])} ..."
                )

        lines.extend(_ambiguity_note(df))
        lines.append("Mẫu dữ liệu:")
        lines.append(df.head(MAX_SAMPLE_ROWS).to_string(index=False))

    return "\n".join(lines)


def format_schemas(frames: dict[str, pd.DataFrame], question_text: str = "") -> str:
    is_multi = len(frames) > 3
    return "\n\n".join(format_schema(var, df, question_text=question_text, is_multi_frame=is_multi) for var, df in frames.items())


from ..normalization.term_mapper import DERIVED_FORMULAS, DERIVED_SUB_TERMS


def format_item_anchors(frames: dict[str, pd.DataFrame], question_text: str, metrics: list[str] | None = None) -> str:
    """Trich xuat top chi tieu khop tu khoa cao nhat trong tung DataFrame va huong dan cong thuc."""
    if not question_text:
        return ""

    stopwords = {
        "bao", "nhiêu", "tổng", "công", "đồng", "triệu", "nghìn", "ngày", "tháng",
        "năm", "trong", "theo", "được", "người", "nhóm", "của", "cho", "vào", "đến", "các", "những"
    }
    q_raw = question_text.lower()
    q_words = [
        w for w in re.sub(r"[^\w\s]", " ", q_raw).split()
        if len(w) >= 2 and w not in stopwords
    ]

    # Mo rong tu khoa voi cac chi tieu phai sinh (Vi du ROE -> them 'loi nhuan sau thue', 'von chu so huu')
    if metrics:
        for m in metrics:
            if m in DERIVED_SUB_TERMS:
                sub_terms = DERIVED_SUB_TERMS[m]
                for st in sub_terms:
                    q_words.extend([w.lower() for w in st.split() if len(w) >= 2 and w.lower() not in stopwords])

    if not q_words:
        return ""

    # Trigram va bigram de bat phrase match
    q_tokens = [w for w in q_raw.split() if w not in stopwords]
    phrases: list[str] = []
    for i in range(len(q_tokens) - 1):
        bg = f"{q_tokens[i]} {q_tokens[i+1]}"
        if len(bg) >= 5:
            phrases.append(bg)
    for i in range(len(q_tokens) - 2):
        tg = f"{q_tokens[i]} {q_tokens[i+1]} {q_tokens[i+2]}"
        if len(tg) >= 8:
            phrases.append(tg)

    anchors: list[str] = []
    var_scores = []
    for var, df in frames.items():
        if "item" not in df.columns:
            continue
        items = df["item"].astype(str).drop_duplicates().tolist()

        scored = []
        for it in items:
            it_clean = re.sub(r"^\d+[\.\)]\s*", "", it).strip()
            it_lower = it_clean.lower()
            
            # 1. Token overlap score
            token_score = sum(1 for w in set(q_words) if w in it_lower)
            if token_score == 0:
                continue
            
            # 2. Phrase match bonus
            phrase_bonus = 0.0
            for phr in phrases:
                if phr in it_lower:
                    phrase_bonus += 4.0
            
            # 3. Concise match bonus / Length penalty: tranh chon dong dai phuc tap khi co dong ngan gon
            length_penalty = len(it_clean) * 0.02
            total_score = token_score + phrase_bonus - length_penalty
            
            scored.append((total_score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored[:2]
        max_s = top_matches[0][0] if top_matches else 0
        var_scores.append((max_s, var, top_matches))

    # Sap xep DataFrame co diem khop cao nhat len dau
    var_scores.sort(key=lambda x: x[0], reverse=True)

    for i, (max_s, var, top_matches) in enumerate(var_scores):
        if top_matches:
            best_tag = " (KHUYẾN NGHỊ DÙNG BẢNG NÀY CHO CHỈ TIÊU NÀY)" if i == 0 and max_s > 0 else ""
            anchors.append(f"GỢI Ý TỪ KHÓA CHỈ TIÊU TRONG `{var}`{best_tag}:")
            for _, it in top_matches:
                short_kw = it.split(" - ")[-1].strip() if " - " in it else it.strip()
                short_kw = re.sub(r"^\d+[\.\)]\s*", "", short_kw)
                short_kw = re.sub(r"\s*\{.*?\}", "", short_kw).strip()
                short_kw = short_kw.replace("'", "\\'").replace("\n", " ").strip()
                if short_kw:
                    anchors.append(f"  * Dùng: `{var}['item'].str.contains('{short_kw}', case=False, na=False, regex=False)`")

    return "\n".join(anchors)


def format_formulas(metrics: list[str]) -> str:
    picked = {m: DERIVED_FORMULAS[m] for m in metrics if m in DERIVED_FORMULAS}
    if not picked:
        picked = DERIVED_FORMULAS
    return "\n".join(f"- {name}: {formula}" for name, formula in picked.items())


_PERIOD_HINTS: dict[str, str] = {
    "opening": (
        "Câu hỏi hỏi số liệu ĐẦU kỳ/đầu năm — lọc `period == \"opening\"`."
    ),
    "closing": (
        "Câu hỏi hỏi số liệu CUỐI kỳ/cuối năm — lọc `period == \"closing\"` "
        "(nếu bảng không có `closing` thì dùng `annual`)."
    ),
    "annual": (
        "Câu hỏi hỏi số phát sinh TRONG năm — dùng `period == \"annual\"` "
        "(nếu bảng không có `annual` thì dùng `closing`)."
    ),
}


def format_period_hint(question: Any) -> str:
    """Goi y chon `period`, suy deterministic tu cau hoi.

    Rong khi cau hoi khong noi ro thoi diem (437/1012 cau) — khi do rule
    mac dinh trong pandas_gen.txt (uu tien closing/annual) da du.
    """
    requested = str(getattr(question, "requested_period", "") or "")
    return _PERIOD_HINTS.get(requested, "")


def format_linked_tables_note(frames: dict[str, pd.DataFrame]) -> str:
    """Xac dinh cac DataFrame thuoc cung 1 bang bi cat trang va tao ghi chu yeu cau ghep."""
    groups: dict[str, list[str]] = {}
    for var, df in frames.items():
        if "group_id" in df.columns and not df.empty:
            gid = str(df["group_id"].iloc[0])
            if gid and gid not in ("None", "nan", ""):
                groups.setdefault(gid, []).append(var)

    notes: list[str] = []
    for gid, vars_list in groups.items():
        if len(vars_list) > 1:
            vars_str = ", ".join(vars_list)
            concat_code = f"df = pd.concat([{vars_str}], ignore_index=True)"
            notes.append(
                f"LƯU Ý ĐẶC BIỆT: Các bảng [{vars_str}] là các phần liên tiếp của cùng 1 bảng báo cáo bị ngắt trang.\n"
                f"BẮT BUỘC bạn phải tự ghép chúng lại trước khi lọc dữ liệu: `{concat_code}`"
            )

    return "\n\n".join(notes)


def format_unit_and_rounding_guidelines(question_text: str, asked_unit: str = "none") -> str:
    """Gợi ý chuẩn hóa đơn vị và làm tròn trong query."""
    q_low = question_text.lower()
    hints = []

    if asked_unit == "ty_dong" or "tỷ đồng" in q_low or "tỉ đồng" in q_low:
        hints.append("YÊU CẦU ĐƠN VỊ: Câu hỏi hỏi TỶ ĐỒNG. Số liệu trong bảng là Đồng (VND) -> BẮT BUỘC chia `1e9` và `result = round(float(sub['value'].iloc[0]) / 1e9, 2)`.")
    elif asked_unit == "trieu_dong" or "triệu đồng" in q_low:
        hints.append("YÊU CẦU ĐƠN VỊ: Câu hỏi hỏi TRIỆU ĐỒNG. Nếu số liệu trong bảng là Đồng (VND, số >= 10^10 hoặc đơn vị vnd) -> chia `1e6` và `result = round(float(sub['value'].iloc[0]) / 1e6, 2)`. Nếu bảng ngân hàng đã là triệu đồng thì giữ nguyên `result = round(float(sub['value'].iloc[0]), 2)`.")
    elif asked_unit == "nghin_dong" or "nghìn đồng" in q_low or "ngàn đồng" in q_low:
        hints.append("YÊU CẦU ĐƠN VỊ: Câu hỏi hỏi NGHÌN ĐỒNG. Chia `1e3` và `result = round(float(sub['value'].iloc[0]) / 1e3, 2)`.")
    elif "cổ phiếu" in q_low or "cổ tức" in q_low:
        if "triệu cổ phiếu" in q_low:
            hints.append("YÊU CẦU ĐƠN VỊ: Hỏi TRIỆU CỔ PHIẾU -> Chia `1e6` và `result = round(float(...) / 1e6, 2)`.")
        elif "nghìn cổ phiếu" in q_low or "ngàn cổ phiếu" in q_low:
            hints.append("YÊU CẦU ĐƠN VỊ: Hỏi NGHÌN CỔ PHIẾU -> Chia `1e3` và `result = round(float(...) / 1e3, 2)`.")
        else:
            hints.append("YÊU CẦU ĐƠN VỊ: Hỏi SỐ LƯỢNG CỔ PHIẾU -> Giữ nguyên số lượng thực: `result = float(...)` hoặc `round(float(...), 2)`.")
    elif any(k in q_low for k in ("tỷ lệ", "tỷ trọng", "phần trăm", "%", "roe", "roa", "biên lợi nhuận", "tăng trưởng")):
        hints.append("YÊU CẦU ĐƠN VỊ: Hỏi TỶ LỆ / % / SUẤT SINH LỜI -> Tính công thức nhân `100.0` và `result = round(float((val1 / val2) * 100.0), 2)`. Nếu trích xuất từ bảng có sẵn %, giữ nguyên.")
    elif any(k in q_low for k in ("năm nào", "vào năm nào", "năm có")):
        hints.append("YÊU CẦU ĐƠN VỊ: Hỏi NĂM -> Gán kết quả dạng số thực của năm: `result = float(max_year)` (ví dụ `2019.0`), tuyệt đối không chia đơn vị tiền.")

    return "\n".join(hints)


def format_candidates(tables: list[RetrievedTable]) -> str:
    return "\n\n".join(
        f"[{t.table_ref}] {t.title or '(không có tiêu đề)'}\n{t.card}" for t in tables
    )


def prompt_dir() -> Path:
    return PROMPT_DIR
