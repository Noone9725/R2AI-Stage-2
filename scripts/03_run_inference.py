"""Stage 2: Inference Pipeline (Ho tro ca 2 pha tach roi va chay tron ven).

Cach dung:
1. Chay Pha 1 (Retrieval doc lap ~1-2 phut):
    python scripts/03_run_inference.py --retrieve-only --questions data/questions/questions.jsonl
    # hoac: python scripts/03_run_inference.py --retrieval

2. Chay Pha 2 (Generation & Execution tu file retrieval):
    python scripts/03_run_inference.py --generate-only --retrieval-input outputs/retrieval/retrieval_results.json
    # hoac: python scripts/03_run_inference.py --generate

3. Chay All (Toan bo 2 pha lien tuc):
    python scripts/03_run_inference.py --all
    # hoac: python scripts/03_run_inference.py --questions data/questions/questions.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

from src.llm.llm_client import LLMClient
from src.pipeline import AnswerPipeline, RetrievalPipeline
from src.utils.io import read_json, read_jsonl, write_json


def load_questions(path: Path) -> list[dict]:
    if str(path).endswith(".jsonl"):
        return list(read_jsonl(path))
    data = read_json(path)
    if isinstance(data, dict):                      # {"questions": [...]}
        data = data.get("questions", data.get("data", []))
    if not isinstance(data, list):
        raise ValueError(f"Khong doc duoc danh sach cau hoi tu {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Chay inference tren bo cau hoi (2 Pha / Toan bo)")
    parser.add_argument("--questions", default=None,
                        help="Duong dan file cau hoi (mac dinh: data/questions/questions.jsonl)")
    parser.add_argument("--out", default=None, help="File predictions/retrieval dau ra")
    parser.add_argument("--limit", type=int, default=None, help="Chi chay N cau dau")
    parser.add_argument("--model", default=None, help="Ghi de llm.model_id")
    parser.add_argument("--backend", default=None, help="vllm | transformers | openai")
    
    # Cac co 2 Pha
    parser.add_argument("--retrieve-only", "--retrieval", dest="retrieve_only", action="store_true",
                        help="Chi chay Pha 1 (Retrieval) -> xuat outputs/retrieval/retrieval_results.json")
    parser.add_argument("--generate-only", "--generate", dest="generate_only", action="store_true",
                        help="Chi chay Pha 2 (Generation & Execution) tu file retrieval co san")
    parser.add_argument("--retrieval-input", "--retrieval-file", dest="retrieval_input", default=None,
                        help="Duong dan file retrieval trung gian cho Phase 2")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Chay toan bo 2 pha lien tuc (mac dinh neu khong chi dinh co pha)")
    args = parser.parse_args()

    settings = bootstrap()

    # 1. Chay Pha 1 (Retrieval Only)
    if args.retrieve_only:
        q_path = Path(args.questions) if args.questions else (
            settings.paths.questions / "questions.jsonl" if (settings.paths.questions / "questions.jsonl").exists()
            else settings.paths.questions / "public_test.json"
        )
        out_r = Path(args.out) if args.out else settings.paths.outputs / "retrieval" / "retrieval_results.json"
        out_r.parent.mkdir(parents=True, exist_ok=True)

        pipeline = RetrievalPipeline()
        pipeline.run(questions_path=q_path, out_path=out_r, limit=args.limit)
        print(f"\n[Phase 1] Da hoan thanh Retrieval -> {out_r}")
        print(f"Meo: Co the dong goi nhanh de kiem thu F2:")
        print(f"  python scripts/04_package.py --pred {out_r} --name retrieval_submission")
        return 0

    # 2. Chay Pha 2 (Generation & Execution Only)
    if args.generate_only:
        r_path = Path(args.retrieval_input) if args.retrieval_input else settings.paths.outputs / "retrieval" / "retrieval_results.json"
        if not r_path.exists():
            print(f"Loi: Khong tim thay file retrieval: {r_path}")
            print("Hay chay voi co `--retrieve-only` truoc de tao file retrieval!")
            return 1

        out_g = Path(args.out) if args.out else settings.paths.outputs / "predictions" / "final_results.json"
        out_g.parent.mkdir(parents=True, exist_ok=True)

        llm = LLMClient(model_id=args.model, backend=args.backend)
        pipeline = AnswerPipeline(llm=llm, load_index=False)
        items = pipeline.run_from_retrieval(
            retrieval_path=r_path,
            out_path=out_g,
            checkpoint_path=out_g,
            save_every=5,
            resume=True,
            limit=args.limit,
        )
        print(f"\n[Phase 2] Da hoan thanh Generation & Execution ({len(items)} cau) -> {out_g}")
        n_code = sum(1 for it in items if it.pandas_query)
        n_ans = sum(1 for it in items if it.answer not in (0.0, None))
        print(f"  co pandas_query : {n_code}/{len(items)}")
        print(f"  answer != 0     : {n_ans}/{len(items)}")
        return 0

    # 3. Chay Toan bo (Mac dinh / All)
    if args.questions:
        q_path = Path(args.questions)
    else:
        default_jsonl = settings.paths.questions / "questions.jsonl"
        default_json = settings.paths.questions / "public_test.json"
        q_path = default_jsonl if default_jsonl.exists() else default_json

    if not q_path.exists():
        print(f"Khong thay file cau hoi: {q_path}")
        return 1

    questions = load_questions(q_path)
    if args.limit:
        questions = questions[: args.limit]
    print(f"Nap {len(questions)} cau hoi tu {q_path}")

    out = Path(args.out) if args.out else settings.paths.outputs / "predictions" / f"{q_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(model_id=args.model, backend=args.backend)
    items = AnswerPipeline(llm=llm).run(
        questions,
        checkpoint_path=out,
        save_every=5,
        resume=True,
    )
    print(f"\nDa ghi {len(items)} ket qua -> {out}")

    n_code = sum(1 for it in items if it.pandas_query)
    n_ans = sum(1 for it in items if it.answer not in (0.0, None))
    print(f"  co pandas_query : {n_code}/{len(items)}")
    print(f"  answer != 0     : {n_ans}/{len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
