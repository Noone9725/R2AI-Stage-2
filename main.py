"""Entry point goi lai cac stage — tuong duong chay lan luot scripts/0*.py.

    python main.py all                                  # corpus -> index -> infer -> package
    python main.py corpus --limit 20
    python main.py index --skip-dense
    python main.py infer --questions data/questions/public_test.json
    python main.py package --pred outputs/predictions/public_test.json
    python main.py eval --pred outputs/predictions/public_test.json

Moi stage cung co the chay doc lap qua scripts/ — main.py chi la vo boc
tien tay cho lan chay day du.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _bootstrap import bootstrap  # noqa: E402

from src.evaluation.run_eval import run_eval  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402
from src.pipeline import AnswerPipeline, CorpusPipeline, IndexPipeline  # noqa: E402
from src.submission import SubmissionPackager  # noqa: E402
from src.utils.io import read_json, write_json  # noqa: E402


def _questions_path(settings, given: str | None) -> Path:
    return Path(given) if given else settings.paths.questions / "public_test.json"


def _load_questions(path: Path) -> list[dict]:
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("questions", data.get("data", []))
    return list(data)


# ── stages ────────────────────────────────────────────────


def stage_corpus(args, settings) -> None:
    stats = CorpusPipeline().run(limit=args.limit, min_rows=None)
    print(f"[corpus] {stats}")


def stage_index(args, settings) -> None:
    stats = IndexPipeline().run(skip_dense=args.skip_dense)
    print(f"[index] {stats}")


def stage_infer(args, settings) -> Path:
    q_path = _questions_path(settings, args.questions)
    questions = _load_questions(q_path)
    if args.limit:
        questions = questions[: args.limit]

    llm = LLMClient(model_id=args.model, backend=args.backend)
    items = AnswerPipeline(llm=llm).run(questions)

    out = settings.paths.outputs / "predictions" / f"{q_path.stem}.json"
    write_json([it.to_dict() for it in items], out)
    print(f"[infer] {len(items)} cau -> {out}")
    return out


def stage_package(args, settings, pred_path: Path | None = None) -> None:
    from importlib import import_module

    to_items = import_module("04_package").to_items

    path = pred_path or Path(args.pred)
    rows = read_json(path)
    q_path = _questions_path(settings, args.questions)
    expected = [int(q["id"]) for q in _load_questions(q_path)] if q_path.exists() else None

    zip_path, report = SubmissionPackager().package(
        to_items(rows), name=args.name, expected_ids=expected, strict=not args.force
    )
    print(f"[package] {zip_path} (ok={report.ok})")


def stage_verify(args, settings) -> int:
    """Kiem tra artifact corpus/index — chay truoc khi infer cho chac."""
    from src.pipeline import check_corpus, check_index

    report = check_index(check_corpus(), require_dense=not args.skip_dense)
    print(report.render())
    return 0 if report.ok else 1


def stage_eval(args, settings) -> None:
    run_eval(
        prediction_path=Path(args.pred),
        gold_path=settings.paths.labels / "gold.json",
        rel_tol=args.rel_tol,
    )


# ── cli ───────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R2AI Stage 2 pipeline")
    parser.add_argument(
        "stage",
        choices=["corpus", "index", "infer", "package", "eval", "verify", "all"],
        help="Stage can chay",
    )
    parser.add_argument("--questions", default=None)
    parser.add_argument("--pred", default=None)
    parser.add_argument("--name", default="submission")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rel-tol", type=float, default=0.01)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = bootstrap()

    if args.stage == "corpus":
        stage_corpus(args, settings)
    elif args.stage == "index":
        stage_index(args, settings)
    elif args.stage == "infer":
        stage_infer(args, settings)
    elif args.stage == "package":
        if not args.pred:
            print("package can --pred")
            return 2
        stage_package(args, settings)
    elif args.stage == "verify":
        return stage_verify(args, settings)
    elif args.stage == "eval":
        if not args.pred:
            print("eval can --pred")
            return 2
        stage_eval(args, settings)
    else:                                   # all
        stage_corpus(args, settings)
        stage_index(args, settings)
        pred = stage_infer(args, settings)
        stage_package(args, settings, pred_path=pred)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
