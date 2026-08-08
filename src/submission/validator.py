"""Kiem tra bai nop TRUOC khi zip.

Moi ngay chi duoc 10 lan nop (public) / 5 lan (private). Mot bai bi
tu choi vi thieu dau "data/" la mat mot luot vo ich. Validator nay chay
cung, bao het loi mot luot chu khong dung o loi dau tien.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REQUIRED_KEYS = (
    "id", "question", "answer", "relevant_docs",
    "relevant_tables", "evidence", "pandas_query",
)


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"LOI ({len(self.errors)}):")
            lines += [f"  x {e}" for e in self.errors[:50]]
            if len(self.errors) > 50:
                lines.append(f"  ... con {len(self.errors) - 50} loi")
        if self.warnings:
            lines.append(f"CANH BAO ({len(self.warnings)}):")
            lines += [f"  ! {w}" for w in self.warnings[:50]]
            if len(self.warnings) > 50:
                lines.append(f"  ... con {len(self.warnings) - 50} canh bao")
        if not lines:
            lines.append("Bai nop hop le.")
        return "\n".join(lines)


def validate_submission(
    items: list[dict[str, Any]],
    root_dir: Path | None = None,
    expected_ids: list[int] | None = None,
    csv_prefix: str = "data/",
) -> ValidationReport:
    """root_dir: thu muc goc cua bai nop (chua data/). None = bo qua check file."""
    rep = ValidationReport()

    if not isinstance(items, list):
        rep.error("submission.json phai la mot list")
        return rep
    if not items:
        rep.error("submission.json rong")
        return rep

    seen_ids: set[int] = set()
    referenced: set[str] = set()

    for idx, item in enumerate(items):
        tag = f"item[{idx}]"
        if not isinstance(item, dict):
            rep.error(f"{tag} khong phai object")
            continue

        for key in _REQUIRED_KEYS:
            if key not in item:
                rep.error(f"{tag} thieu key '{key}'")

        qid = item.get("id")
        if not isinstance(qid, int) or isinstance(qid, bool):
            rep.error(f"{tag} id phai la int, dang la {type(qid).__name__}")
        else:
            tag = f"id={qid}"
            if qid in seen_ids:
                rep.error(f"{tag} bi trung")
            seen_ids.add(qid)

        if not isinstance(item.get("question"), str) or not item.get("question"):
            rep.error(f"{tag} question phai la chuoi khong rong")

        answer = item.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (int, float)):
            rep.error(f"{tag} answer phai la so, dang la {type(answer).__name__}")
        elif answer != answer or answer in (float("inf"), float("-inf")):
            rep.error(f"{tag} answer la NaN/inf — JSON khong hop le")

        docs = item.get("relevant_docs")
        if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
            rep.error(f"{tag} relevant_docs phai la list[str]")
        elif not docs:
            rep.warn(f"{tag} relevant_docs rong — mat diem recall")
        else:
            for d in docs:
                if d.endswith(".txt"):
                    rep.error(f"{tag} relevant_docs con duoi .txt: {d}")

        tables = item.get("relevant_tables")
        if not isinstance(tables, list) or not all(isinstance(t, str) for t in tables):
            rep.error(f"{tag} relevant_tables phai la list[str]")
        elif not tables:
            rep.warn(f"{tag} relevant_tables rong — mat diem recall")
        else:
            for t in tables:
                if "|" not in t:
                    rep.error(f"{tag} relevant_tables sai dinh dang (can '<doc_id>|<position>'): {t}")
                else:
                    doc_part, _, pos = t.rpartition("|")
                    if not pos.strip().lstrip("-").isdigit():
                        rep.error(f"{tag} position khong phai so: {t}")
                    if isinstance(docs, list) and docs and doc_part not in docs:
                        rep.warn(f"{tag} table '{t}' co doc_id khong nam trong relevant_docs")

        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            rep.error(f"{tag} evidence phai la list")
        else:
            for ev in evidence:
                if not isinstance(ev, dict) or "variable" not in ev or "csv_path" not in ev:
                    rep.error(f"{tag} evidence phai la {{variable, csv_path}}")
                    continue
                path = str(ev["csv_path"])
                if "\\" in path:
                    rep.error(f"{tag} csv_path dung dau '\\': {path}")
                if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
                    rep.error(f"{tag} csv_path phai la duong dan tuong doi: {path}")
                if not path.startswith(csv_prefix):
                    rep.error(f"{tag} csv_path phai bat dau bang '{csv_prefix}': {path}")
                referenced.add(path)

            code = item.get("pandas_query", "")
            if isinstance(code, str) and code:
                for ev in evidence:
                    if isinstance(ev, dict) and str(ev.get("variable", "")) not in code:
                        rep.warn(f"{tag} bien '{ev.get('variable')}' khong xuat hien trong pandas_query")

        code = item.get("pandas_query")
        if not isinstance(code, str):
            rep.error(f"{tag} pandas_query phai la chuoi")
        elif not code.strip():
            rep.warn(f"{tag} pandas_query rong — mat diem Execution Accuracy")
        elif "result" not in code:
            rep.warn(f"{tag} pandas_query khong gan bien `result`")

    if expected_ids is not None:
        missing = sorted(set(expected_ids) - seen_ids)
        extra = sorted(seen_ids - set(expected_ids))
        if missing:
            rep.error(f"Thieu {len(missing)} cau hoi, vd id: {missing[:10]}")
        if extra:
            rep.error(f"Co {len(extra)} id la, vd: {extra[:10]}")

    if root_dir is not None:
        for path in sorted(referenced):
            if not (root_dir / path).exists():
                rep.error(f"CSV duoc tham chieu nhung khong co file: {path}")

    return rep
