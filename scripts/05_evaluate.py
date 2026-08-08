"""Danh gia local tren nhan tay.

BTC khong phat dev set — muon biet dang o dau thi phai tu gan nhan
~50-100 cau vao labels/gold.json roi do lien tuc.

Dinh dang gold (list):
    {"id": 1, "answer": 123.45,
     "relevant_docs": ["AAA_..._2020_consolidated"],
     "relevant_tables": ["AAA_..._2020_consolidated|3"]}

    python scripts/05_evaluate.py --pred outputs/predictions/public_test.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

from src.evaluation.run_eval import run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval local tren nhan tay")
    parser.add_argument("--pred", required=True, help="File predictions JSON")
    parser.add_argument("--gold", default=None, help="Mac dinh: labels/gold.json")
    parser.add_argument("--out", default=None, help="File bao cao JSON dau ra")
    parser.add_argument("--rel-tol", type=float, default=0.01,
                        help="Sai so tuong doi coi la dung (BTC chua cong bo)")
    parser.add_argument("--worst", type=int, default=15, help="So cau te nhat in ra")
    args = parser.parse_args()

    settings = bootstrap()
    gold = Path(args.gold) if args.gold else settings.paths.labels / "gold.json"
    if not gold.exists():
        print(f"Chua co file nhan: {gold}\n"
              "Tao labels/gold.json voi it nhat vai chuc cau da gan nhan tay.")
        return 1

    report = run_eval(
        prediction_path=Path(args.pred),
        gold_path=gold,
        out_path=Path(args.out) if args.out else None,
        rel_tol=args.rel_tol,
        show_worst=args.worst,
    )
    return 0 if report.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
