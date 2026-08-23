"""Chan doan loi thuc thi -> goi y cu the cho vong self-repair.

LLM sua code tot hon nhieu khi duoc chi ro "cot 'nam' khong ton tai, cot
that la 'year'" thay vi chi nhan KeyError. Ham nay bien traceback thanh
goi y hanh dong.
"""

from __future__ import annotations

import re

import pandas as pd

from ..schemas import ExecutionResult

_KEY_RE = re.compile(r"['\"]([^'\"]+)['\"]")


def diagnose(result: ExecutionResult, frames: dict[str, pd.DataFrame]) -> str:
    hints: list[str] = []
    err = result.error
    etype = result.error_type

    if etype == "KeyError":
        missing = _KEY_RE.findall(err)
        for key in missing:
            hints.append(f"- Khoá '{key}' không tồn tại.")
        cols = sorted({c for df in frames.values() for c in df.columns})
        hints.append(f"- Các cột thực có: {cols}")
        hints.append("- Nếu đang lọc tên chỉ tiêu, hãy dùng cột `item` với .str.contains(...).")

    elif etype == "IndexError":
        hints.append("- Bộ lọc trả về 0 dòng nên .iloc[0] lỗi.")
        hints.append("- Nới lỏng điều kiện: dùng từ khoá ngắn hơn, case=False, na=False.")
        for var, df in frames.items():
            if "item" in df.columns:
                sample = df["item"].astype(str).drop_duplicates().head(15).tolist()
                hints.append(f"- Giá trị `item` có trong {var}: {sample}")

    elif etype == "NameError":
        names = _KEY_RE.findall(err) or re.findall(r"name (\w+)", err)
        hints.append(f"- Biến chưa được định nghĩa: {names}.")
        hints.append(f"- Chỉ được dùng các biến: {sorted(frames)}")

    elif etype == "AttributeError":
        hints.append("- Phương thức không tồn tại trên đối tượng đó.")
        hints.append("- Kiểm tra bạn đang gọi trên DataFrame hay Series.")

    elif etype == "ZeroDivisionError":
        hints.append("- Mẫu số bằng 0. Kiểm tra lại dòng dữ liệu lấy làm mẫu số.")

    elif etype == "ResultTypeError":
        hints.append(f"- {err}")
        hints.append("- Thêm .iloc[0] hoặc .sum() để lấy đúng một số, rồi float(...).")

    elif etype == "UnsafeCodeError":
        hints.append(f"- {err}")
        hints.append("- Không import, không đọc file. Chỉ dùng biến DataFrame đã cấp và pd/np.")

    elif etype == "EmptyFilterError":
        hints.append(f"- {err}")
        hints.append("- Bộ lọc trả về rỗng (sub.empty) do từ khóa hoặc điều kiện không khớp dòng nào.")
        contains_match = re.findall(r"\.str\.contains\(\s*['\"]([^'\"]+)['\"]", result.code)
        found_in_any = False
        if contains_match:
            kw = contains_match[0]
            for var, df in frames.items():
                if "item" in df.columns:
                    m = df[df["item"].astype(str).str.contains(kw, case=False, na=False, regex=False)]
                    if not m.empty:
                        hints.append(f"- CHÚ Ý: Từ khóa '{kw}' ĐƯỢC TÌM THẤY TRONG `{var}`! Hãy đổi code sang lọc trên `{var}`.")
                        found_in_any = True

        if not found_in_any:
            hints.append("- Từ khóa trên KHÔNG TỒN TẠI trong bất kỳ DataFrame nào! Hãy chọn lại từ khóa từ danh sách `item` thực tế:")
            for var, df in frames.items():
                if "item" in df.columns:
                    sample = df["item"].astype(str).drop_duplicates().head(10).tolist()
                    hints.append(f"  * Các `item` thực tế trong `{var}`: {sample}")

    elif etype == "TimeoutError":
        hints.append("- Code chạy quá lâu, có thể do vòng lặp. Dùng phép vector hoá của pandas.")

    elif etype == "SyntaxError":
        hints.append("- Code sai cú pháp. Viết lại toàn bộ, gọn và đúng thụt lề.")

    if not hints:
        hints.append(f"- Lỗi: {err}. Viết lại code theo cách đơn giản hơn.")

    return "\n".join(hints)


def validate_query_text(code: str) -> list[str]:
    """Canh bao tinh (khong chan) — dung de log chat luong generation."""
    warnings: list[str] = []
    if "read_csv" in code:
        warnings.append("code tu doc CSV thay vi dung bien co san")
    if "==" in code and ".str.contains" not in code:
        warnings.append("dung so sanh bang tuyet doi cho ten chi tieu (de sai dau)")
    if "print(" in code and "result" not in code.split("print(")[0]:
        warnings.append("chi print, chua chac gan result")
    return warnings
