"""Stage 2: cau hoi -> submission.json (chua dong goi).

    python scripts/03_run_inference.py --questions data/questions/public_test.json
    python scripts/03_run_inference.py --limit 5 --backend transformers

Ket qua ghi ra outputs/predictions/<name>.json — dau vao cho 04_package
va 05_evaluate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

from src.llm.llm_client import LLMClient
from src.pipeline import AnswerPipeline
from src.utils.io import read_json, write_json


def load_questions(path: Path) -> list[dict]:
    data = read_json(path)
    if isinstance(data, dict):                      # {"questions": [...]}
        data = data.get("questions", data.get("data", []))
    if not isinstance(data, list):
        raise ValueError(f"Khong doc duoc danh sach cau hoi tu {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Chay inference tren bo cau hoi")
    parser.add_argument("--questions", default=None,
                        help="Duong dan file cau hoi (mac dinh: data/questions/public_test.json)")
    parser.add_argument("--out", default=None, help="File predictions dau ra")
    parser.add_argument("--limit", type=int, default=None, help="Chi chay N cau dau")
    parser.add_argument("--model", default=None, help="Ghi de llm.model_id")
    parser.add_argument("--backend", default=None, help="vllm | transformers | openai")
    args = parser.parse_args()

    settings = bootstrap()

    q_path = Path(args.questions) if args.questions else settings.paths.questions / "public_test.json"
    if not q_path.exists():
        print(f"Khong thay file cau hoi: {q_path}")
        return 1

    questions = load_questions(q_path)
    if args.limit:
        questions = questions[: args.limit]
    print(f"Nap {len(questions)} cau hoi tu {q_path}")

    llm = LLMClient(model_id=args.model, backend=args.backend)
    items = AnswerPipeline(llm=llm).run(questions)

    out = Path(args.out) if args.out else settings.paths.outputs / "predictions" / f"{q_path.stem}.json"
    write_json([it.to_dict() for it in items], out)
    print(f"\nDa ghi {len(items)} ket qua -> {out}")

    n_code = sum(1 for it in items if it.pandas_query)
    n_ans = sum(1 for it in items if it.answer not in (0.0, None))
    print(f"  co pandas_query : {n_code}/{len(items)}")
    print(f"  answer != 0     : {n_ans}/{len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
