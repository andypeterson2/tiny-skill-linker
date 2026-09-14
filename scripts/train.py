"""Fine-tune a sentence embedder on TechWolf's synthetic ESCO skill sentences.

The defaults follow Decorte et al. (2023), Appendix C: MultipleNegativesRankingLoss with
scale 20, one epoch, batches of 64, AdamW at 2e-5 with linear warmup over 5% of steps, and
each sentence joined to one random other sentence before or after it.

Usage:
    python scripts/train.py --seed 0 --out models/ft-seed0
    python scripts/train.py --seed 0 --holdout-frac 0.2 --out models/ft-holdout-seed0
"""

import argparse
import json
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

ROOT = Path(__file__).resolve().parent.parent
DATASET = "TechWolf/Synthetic-ESCO-skill-sentences"


def augment(sentences: list[str], rng: random.Random) -> list[str]:
    out = []
    for i, s in enumerate(sentences):
        j = rng.randrange(len(sentences) - 1)
        other = sentences[j if j < i else j + 1]
        out.append(f"{s} {other}" if rng.random() < 0.5 else f"{other} {s}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="for timing runs only")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_dataset(DATASET, split="train")
    skills = sorted(set(rows["skill"]))
    heldout = set(rng.sample(skills, round(len(skills) * args.holdout_frac)))
    rows = rows.filter(lambda r: r["skill"] not in heldout)

    sentences = rows["sentence"]
    anchors = sentences if args.no_augment else augment(sentences, rng)
    train = Dataset.from_dict({"anchor": anchors, "positive": rows["skill"]}).shuffle(
        seed=args.seed
    )

    model = SentenceTransformer(args.base, device="cpu")
    loss = MultipleNegativesRankingLoss(model, scale=20.0)
    out = ROOT / args.out
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        # The same skill twice in a batch would score its own positive as a negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=args.seed,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
    )
    start = time.time()
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=train, loss=loss).train()
    elapsed = time.time() - start
    model.save_pretrained(str(out))

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    record = {
        "base": args.base,
        "out": args.out,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "augment": not args.no_augment,
        "holdout_frac": args.holdout_frac,
        "n_train_pairs": len(train),
        "n_skills_trained": len(skills) - len(heldout),
        "heldout_skills": sorted(heldout),
        "provenance": {
            "dataset": DATASET,
            "dataset_revision": HfApi().repo_info(DATASET, repo_type="dataset").sha,
            "code_sha": sha.stdout.strip(),
            "versions": {p: version(p) for p in ("sentence-transformers", "transformers", "torch")},
            "seconds": round(elapsed, 1),
        },
    }
    dest = ROOT / "results" / "train" / f"{Path(args.out).name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n")
    print(f"trained {len(train)} pairs in {elapsed / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
