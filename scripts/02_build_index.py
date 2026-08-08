"""Stage 1b: manifest.jsonl -> BM25 index + dense vector index.

Tach khoi stage 1: doi model embedding hay doi cach viet card thi chi can
chay lai file nay, khong phai extract lai tu .txt.

    python scripts/02_build_index.py
    python scripts/02_build_index.py --skip-dense    # chi BM25 (khong can GPU)
"""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

from src.embeddings.embedder import Embedder
from src.pipeline import IndexPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 + dense index")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Chi build BM25, bo qua embedding")
    parser.add_argument("--device", default=None, help="cpu | cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    bootstrap()
    embedder = Embedder(device=args.device, batch_size=args.batch_size)
    stats = IndexPipeline(embedder=embedder).run(skip_dense=args.skip_dense)

    print("\n=== Stage 1b ===")
    for key, value in stats.items():
        print(f"  {key:20s} {value}")
    if stats["tables"] == 0:
        print("\nManifest rong — chay scripts/01_build_corpus.py truoc.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
