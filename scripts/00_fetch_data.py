"""Tai bo du lieu ViFinQA tu HuggingFace ve data/.

    python scripts/00_fetch_data.py --questions-only   # chi lay questions + code_stock
    python scripts/00_fetch_data.py                    # + toan bo corpus (~363 MiB)
    python scripts/00_fetch_data.py --tickers AAA,VNM  # chi mot so ma

Nguon (BTC cung cap, xem competition_documents/04.data.md):
    https://huggingface.co/datasets/AIGuruTinix/ViFinQA
Corpus goc: TiniX Vietnam OCR Annual Financial Statements — CC BY-NC 4.0.

Token doc tu HF_TOKEN trong .env (khong bao gio in ra).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from _bootstrap import bootstrap

REPO_ID = "AIGuruTinix/ViFinQA"
SMALL_FILES = {
    "questions/questions.jsonl": "data/questions/questions.jsonl",
    "code_stock.csv": "data/questions/code_stock.csv",
}


def _token() -> str | None:
    tok = os.getenv("HF_TOKEN")
    return tok.strip() or None if tok else None


def fetch_small(root: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    for remote, local in SMALL_FILES.items():
        dst = root / local
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download(REPO_ID, remote, repo_type="dataset", token=token)
        shutil.copy(src, dst)
        print(f"[fetch] {local}  ({dst.stat().st_size:,} bytes)")


def fetch_corpus(root: Path, token: str | None, tickers: list[str] | None) -> None:
    """Tai corpus. HF tra 429 khi keo 1973 file song song, nen:
    - max_workers thap (4)
    - retry nhieu vong voi backoff; snapshot_download tu bo qua file da co
    """
    import time

    from huggingface_hub import snapshot_download

    patterns = (
        [f"financial_statements/{t}/**" for t in tickers]
        if tickers
        else ["financial_statements/**"]
    )
    dest = root / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] corpus -> {dest} (patterns: {len(patterns)})")

    delay = 10
    for attempt in range(1, 9):
        try:
            snapshot_download(
                REPO_ID,
                repo_type="dataset",
                token=token,
                allow_patterns=patterns,
                local_dir=dest,
                max_workers=4,
                tqdm_class=None,
            )
            break
        except Exception as exc:  # noqa: BLE001 — 429/mang chap chon la binh thuong
            n = sum(1 for _ in dest.rglob("*_extracted.txt"))
            print(f"[fetch] vong {attempt} loi ({type(exc).__name__}); da co {n} file; cho {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    n = sum(1 for _ in dest.rglob("*_extracted.txt"))
    print(f"[fetch] done — {n} file .txt trong {dest}")


def run_fetch(questions_only: bool = False, tickers: list[str] | None = None) -> None:
    root = bootstrap().root
    token = _token()
    print(f"[fetch] HF_TOKEN: {'co' if token else 'khong co (repo public van tai duoc)'}")

    fetch_small(root, token)
    if not questions_only:
        fetch_corpus(root, token, tickers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions-only", action="store_true")
    ap.add_argument("--tickers", help="danh sach ma CK, phan cach dau phay")
    args = ap.parse_args(argv)

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else None
    )
    run_fetch(questions_only=args.questions_only, tickers=tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
