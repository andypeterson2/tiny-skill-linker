"""General-retrieval check: NanoBEIR NDCG@10 for one model, written to results/nanobeir/.

NanoBEIR is 13 small BEIR subsets (about 50 queries each). A fine-tuned model that
loses much here has given up general ability for the skill-linking task.

Usage:
    python scripts/nanobeir.py --model models/ft-seed1 --label ft-seed1
"""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator

from tiny_skill_linker.provenance import code_state, weights_digest

ROOT = Path(__file__).resolve().parent.parent
# NanoBEIREvaluator reads all 13 subsets from one dataset repository.
DATASET = "sentence-transformers/NanoBEIR-en"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    model = SentenceTransformer(args.model, device="cpu")
    start = time.time()
    scores = NanoBEIREvaluator(batch_size=64, show_progress_bar=False)(model)
    elapsed = time.time() - start
    ndcg = {k: v for k, v in scores.items() if k.endswith("cosine_ndcg@10")}
    subsets = sorted(
        k.removeprefix("Nano").removesuffix("_cosine_ndcg@10") for k in ndcg if "mean" not in k
    )
    out = {
        "model": args.model,
        "label": args.label,
        "ndcg@10": ndcg,
        "provenance": {
            "weights_digest": weights_digest(args.model),
            "dataset": DATASET,
            "dataset_revision": HfApi().repo_info(DATASET, repo_type="dataset").sha,
            "subsets": subsets,
            "code": code_state(),
            "versions": {
                p: version(p) for p in ("sentence-transformers", "transformers", "torch")
            },
            "machine": platform.platform(),
            "seconds": round(elapsed, 1),
        },
    }
    dest = ROOT / "results" / "nanobeir" / f"{args.label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    mean = next(v for k, v in ndcg.items() if "mean" in k)
    print(f"{args.label}: NanoBEIR mean NDCG@10 {mean:.4f}")


if __name__ == "__main__":
    main()
