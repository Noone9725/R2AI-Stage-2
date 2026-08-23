"""Doc/ghi JSON, JSONL, CSV, pickle — dung chung toan pipeline."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_json_atomic(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_atomic(rows: Iterable[dict], path: str | Path) -> int:
    """Ghi JSONL an toan: temp -> os.replace (fallback sang ghi truc tiep neu FUSE loi)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rows_list = list(rows) if not isinstance(rows, list) else rows
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        os.replace(tmp, p)
        return len(rows_list)
    except (OSError, PermissionError):
        # Fallback truc tiep khi filesystem khong ho tro atomic replace (Google Drive FUSE)
        with p.open("w", encoding="utf-8") as f:
            for row in rows_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        return len(rows_list)


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """Doc file .txt OCR — fallback sang latin-1 neu encoding loi."""
    p = Path(path)
    try:
        return p.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1", errors="replace")


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def save_pickle(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)
