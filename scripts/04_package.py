"""Stage 3: predictions.json -> ZIP nop bai (co validate truoc khi zip).

    python scripts/04_package.py --pred outputs/predictions/public_test.json
    python scripts/04_package.py --pred ... --name run_2026_08_03
    python scripts/04_package.py --pred ... --check-only

Quota nop rat it (10 lan/ngay public, 5 lan tong private) — mac dinh
strict=True: co bat ky loi schema nao thi KHONG zip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

from src.schemas import Evidence, SubmissionItem
from src.submission import SubmissionPackager, validate_submission
from src.utils.io import read_json


def to_items(rows: list[dict]) -> list[SubmissionItem]:
    return [
        SubmissionItem(
            id=int(r["id"]),
            question=str(r.get("question", "")),
            answer=float(r.get("answer") or 0.0),
            relevant_docs=list(r.get("relevant_docs") or []),
            relevant_tables=list(r.get("relevant_tables") or []),
            evidence=[
                Evidence(variable=str(e["variable"]), csv_path=str(e["csv_path"]))
                for e in (r.get("evidence") or [])
            ],
            pandas_query=str(r.get("pandas_query") or ""),
        )
        for r in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dong goi bai nop")
    parser.add_argument("--pred", required=True, help="File predictions JSON")
    parser.add_argument("--name", default="submission", help="Ten thu muc stage + file ZIP")
    parser.add_argument("--questions", default=None,
                        help="File cau hoi goc — de doi chieu thieu/thua id")
    parser.add_argument("--check-only", action="store_true", help="Chi validate, khong zip")
    parser.add_argument("--force", action="store_true", help="Zip du co loi (khong khuyen khich)")
    args = parser.parse_args()

    if args.name.strip("/ ") in {"", "submissions", "predictions"}:
        print("--name khong duoc la 'submissions'/'predictions' (se xoa nham thu muc).")
        return 2

    bootstrap()
    rows = read_json(Path(args.pred))
    items = to_items(rows)
    print(f"Nap {len(items)} ket qua tu {args.pred}")

    expected_ids = None
    if args.questions:
        qpath = Path(args.questions)
        if str(qpath).endswith(".jsonl"):
            from src.utils.io import read_jsonl
            qs = list(read_jsonl(qpath))
        else:
            qs = read_json(qpath)
            if isinstance(qs, dict):
                qs = qs.get("questions", qs.get("data", []))
        expected_ids = [int(q["id"]) for q in qs]

    packager = SubmissionPackager()

    if args.check_only:
        report = validate_submission(rows, expected_ids=expected_ids)
        print(report.render())
        return 0 if report.ok else 1

    zip_path, report = packager.package(
        items, name=args.name, expected_ids=expected_ids, strict=not args.force
    )
    print(f"\nSan sang nop: {zip_path}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
