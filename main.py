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
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _bootstrap import bootstrap  # noqa: E402

from src.evaluation.run_eval import run_eval  # noqa: E402
from src.llm.llm_client import LLMClient  # noqa: E402
from src.pipeline import AnswerPipeline, CorpusPipeline, IndexPipeline  # noqa: E402
from src.submission import SubmissionPackager  # noqa: E402
from src.utils.io import read_json, write_json  # noqa: E402


def _questions_path(settings, given: str | None) -> Path:
    if given:
        p = Path(given)
        if p.exists():
            return p
    default_jsonl = settings.paths.questions / "questions.jsonl"
    if default_jsonl.exists():
        return default_jsonl
    default_json = settings.paths.questions / "public_test.json"
    if default_json.exists():
        return default_json
    # Tu dong fetch questions neu chua co
    try:
        from importlib import import_module
        run_fetch = import_module("00_fetch_data").run_fetch
        run_fetch(questions_only=True)
    except Exception:
        pass
    return default_jsonl if default_jsonl.exists() else default_json


def _load_questions(path: Path) -> list[dict]:
    if str(path).endswith(".jsonl"):
        from src.utils.io import read_jsonl
        return list(read_jsonl(path))
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("questions", data.get("data", []))
    return list(data)


def stage_fetch(args, settings) -> None:
    from importlib import import_module
    run_fetch = import_module("00_fetch_data").run_fetch
    run_fetch(questions_only=getattr(args, "questions_only", False))


def stage_corpus(args, settings) -> None:
    stats = CorpusPipeline().run(
        limit=args.limit,
        min_rows=None,
        resume=not getattr(args, "no_resume", False),
    )
    print(f"[corpus] {stats}")


def stage_index(args, settings) -> None:
    stats = IndexPipeline().run(
        skip_dense=args.skip_dense,
        force_manifest_rebuild=getattr(args, "rebuild_manifest", False),
    )
    print(f"[index] {stats}")


def stage_infer(args, settings) -> Path:
    q_path = _questions_path(settings, args.questions)
    questions = _load_questions(q_path)
    if args.limit:
        questions = questions[: args.limit]

    out = Path(args.pred) if args.pred else settings.paths.outputs / "predictions" / f"{q_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(model_id=args.model, backend=args.backend)
    items = AnswerPipeline(llm=llm).run(
        questions,
        checkpoint_path=out,
        save_every=5,
        resume=True,
    )

    print(f"[infer] {len(items)} cau -> {out}")
    return out


def stage_package(args, settings, pred_path: Path | None = None) -> None:
    from importlib import import_module

    to_items = import_module("04_package").to_items

    path = pred_path or Path(args.pred)
    rows = read_json(path)
    q_path = _questions_path(settings, args.questions)
    expected = [int(q["id"]) for q in _load_questions(q_path)] if q_path.exists() else None
    if expected and getattr(args, "limit", None):
        expected = expected[: args.limit]

    zip_path, report = SubmissionPackager().package(
        to_items(rows), name=args.name, expected_ids=expected, strict=not args.force
    )
    if getattr(args, "out", None):
        out_dst = Path(args.out)
        out_dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(zip_path, out_dst)
        print(f"[package] Da luu them ban copy tai {out_dst}")
    print(f"[package] {zip_path} (ok={report.ok})")


def stage_verify(args, settings) -> int:
    """Kiem tra artifact corpus/index — chay truoc khi infer cho chac."""
    from src.pipeline import check_corpus, check_index

    report = check_index(check_corpus(), require_dense=not args.skip_dense)
    print(report.render())
    return 0 if report.ok else 1


def stage_eval(args, settings) -> None:
    gold_path = Path(args.gold) if getattr(args, "gold", None) else settings.paths.labels / "gold.json"
    if not gold_path.exists():
        print(f"Khong tim thay file gold label tai: {gold_path}")
        return
    run_eval(
        prediction_path=Path(args.pred),
        gold_path=gold_path,
        out_path=Path(args.out) if getattr(args, "out", None) else None,
        rel_tol=args.rel_tol,
        show_worst=getattr(args, "worst", 15),
    )


# ── cli ───────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R2AI Stage 2 pipeline")
    parser.add_argument(
        "stage",
        choices=["fetch", "corpus", "index", "infer", "package", "eval", "evaluate", "verify", "all"],
        help="Stage can chay",
    )
    parser.add_argument("--fetch", action="store_true", help="Tu dong fetch data tu HuggingFace truoc khi chay")
    parser.add_argument("--no-resume", action="store_true", help="Khong dung checkpoint resume trong corpus")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Tu dong scan va tao lai manifest.jsonl tu data/processed/")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--pred", default=None)
    parser.add_argument("--gold", default=None, help="Duong dan file gold labels JSON")
    parser.add_argument("--name", default="submission")
    parser.add_argument("--out", default=None, help="Duong dan file dau ra (ZIP hoac JSON report)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rel-tol", type=float, default=0.01)
    parser.add_argument("--worst", type=int, default=15, help="So luong cau sai in ra chi tiet")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = bootstrap()

    if args.stage == "fetch":
        stage_fetch(args, settings)
    elif args.stage == "corpus":
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
    elif args.stage in ("eval", "evaluate"):
        if not args.pred:
            print("eval can --pred")
            return 2
        stage_eval(args, settings)
    else:                                   # all
        # 1. Kiem tra data/raw
        raw_files = list(settings.paths.raw.rglob("*.txt")) if settings.paths.raw.exists() else []
        if args.fetch or not raw_files:
            print("[all] Khong thay data/raw hoac co co --fetch -> Tien hanh fetch data...")
            stage_fetch(args, settings)
        else:
            print(f"[all] Tim thay {len(raw_files)} file raw .txt -> Bo qua stage_fetch.")

        # 2. Kiem tra data/processed
        processed_files = list(settings.paths.processed.glob("*.csv")) if settings.paths.processed.exists() else []
        if not processed_files:
            print("[all] data/processed rong -> Tien hanh chay stage_corpus...")
            stage_corpus(args, settings)
        else:
            print(f"[all] Tim thay {len(processed_files)} bang CSV trong data/processed -> Bo qua stage_corpus.")

        # 3. Kiem tra data/index
        bm25_file = settings.paths.index / "bm25.pkl"
        manifest_file = settings.paths.index / "manifest.jsonl"
        if not bm25_file.exists() or not manifest_file.exists():
            print("[all] data/index thieu bm25.pkl hoac manifest -> Tien hanh chay stage_index...")
            stage_index(args, settings)
        else:
            print("[all] Tim thay chi muc trong data/index -> Bo qua stage_index.")

        # 4. Chay infer va package
        pred = stage_infer(args, settings)
        stage_package(args, settings, pred_path=pred)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
