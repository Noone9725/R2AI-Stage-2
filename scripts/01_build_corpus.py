"""Stage 1: data/raw/**/*.txt -> data/processed/*.csv + data/index/manifest.jsonl.

Chay MOT LAN cho ca cuoc thi (hoac khi doi tham so extraction).

    python scripts/01_build_corpus.py                # toan bo kho
    python scripts/01_build_corpus.py --limit 20     # thu nhanh tren 20 file
"""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

from src.pipeline import CorpusPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corpus CSV + manifest")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chi xu ly N file dau (de thu nhanh)")
    parser.add_argument("--min-rows", type=int, default=None,
                        help="Bo bang it hon N dong (mac dinh: corpus.min_table_rows)")
    args = parser.parse_args()

    bootstrap()
    stats = CorpusPipeline().run(limit=args.limit, min_rows=args.min_rows)

    print("\n=== Stage 1 ===")
    for key, value in stats.items():
        print(f"  {key:20s} {value}")
    if stats["tables_written"] == 0:
        print("\nCanh bao: khong ghi duoc bang nao — kiem tra data/raw/ va table_detector.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
