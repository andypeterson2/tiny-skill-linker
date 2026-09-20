"""Check the synthetic training sentences against the evaluation query sentences.

The fine-tuning gain is only meaningful if no training sentence appears in a test set.
This reports exact and normalised overlap for every split the project evaluates on.

Usage:
    python scripts/overlap.py
"""

import json
import platform
import re
import time
from pathlib import Path

from datasets import load_dataset

from tiny_skill_linker.provenance import code_state
from tiny_skill_linker.tasks import ESCO_VERSION, TASKS, VAL_TASKS, load_task

ROOT = Path(__file__).resolve().parent.parent
DATASET = "TechWolf/Synthetic-ESCO-skill-sentences"


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def main() -> None:
    start = time.time()
    rows = load_dataset(DATASET, split="train")
    train = set(rows["sentence"])
    train_normalised = {normalise(s) for s in train}

    splits = {}
    for split in ("test", "val"):
        names = [n for n in TASKS if split == "test" or n in VAL_TASKS]
        for name in names:
            queries = load_task(name, split).datasets["en"].query_texts
            exact = sorted(set(queries) & train)
            loose = sorted({normalise(q) for q in queries} & train_normalised)
            splits[f"{split}/{name}"] = {
                "n_queries": len(queries),
                "n_exact_overlap": len(exact),
                "n_normalised_overlap": len(loose),
                "examples": exact[:5],
            }
            print(
                f"{split}/{name:9s} queries {len(queries):4d}  exact {len(exact):3d}"
                f"  normalised {len(loose):3d}"
            )

    out = {
        "train_dataset": DATASET,
        "n_train_sentences": len(train),
        "splits": splits,
        "provenance": {
            "esco_version": ESCO_VERSION,
            "code": code_state(),
            "machine": platform.platform(),
            "seconds": round(time.time() - start, 1),
        },
    }
    dest = ROOT / "results" / "overlap.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
