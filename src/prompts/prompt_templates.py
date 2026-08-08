"""Load + render prompt template tu configs/prompts/*.txt.

Tach prompt ra file text de sua khong can sua code — vong lap tuning
prompt la vong lap chay nhieu nhat trong cuoc thi nay.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
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

MAX_SAMPLE_ROWS = 8
MAX_ITEM_PREVIEW = 25
MAX_COLUMN_LABELS = 12
MAX_AMBIGUOUS_ITEMS = 5


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


def format_schema(var: str, df: pd.DataFrame) -> str:
    """Lieu ke cot + mau dong + danh sach `item` co san.

    Danh sach `item` la phan quan trong nhat: LLM khong doan duoc ten chi
    tieu tieng Viet co dau/khong dau, thay vao do nen copy tu day.
    """
    lines = [f"### {var}", f"Số dòng: {len(df)}", f"Cột: {list(df.columns)}"]

    if "item" in df.columns:
        items = df["item"].astype(str).drop_duplicates().tolist()
        shown = items[:MAX_ITEM_PREVIEW]
        lines.append("Các giá trị cột `item`:")
        lines.extend(f"  - {it}" for it in shown)
        if len(items) > len(shown):
            lines.append(f"  ... còn {len(items) - len(shown)} chỉ tiêu khác")

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
        # Bang truc category ("Nha cua | May moc | ..."): day la chieu thu hai
        # phan biet cac dong cung `item`. Bang truc thoi gian thi nhan trung
        # voi source_date nen khong can nhac lai.
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


def format_schemas(frames: dict[str, pd.DataFrame]) -> str:
    return "\n\n".join(format_schema(var, df) for var, df in frames.items())


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


def format_candidates(tables: list[RetrievedTable]) -> str:
    return "\n\n".join(
        f"[{t.table_ref}] {t.title or '(không có tiêu đề)'}\n{t.card}" for t in tables
    )


def prompt_dir() -> Path:
    return PROMPT_DIR
