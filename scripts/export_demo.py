"""Write the ranked candidates the website's benchmark viewer reads.

The published results carry only aggregates and, per query, each gold skill's rank.
Neither says what sits at ranks 0-9, so a reader cannot see what the model actually
returned. This re-encodes the test split and keeps the top candidates per query,
which is enough for a page to recompute RP@5 from rows a visitor can scroll.

Usage:
    python scripts/export_demo.py --out ../website/public/skill-linker/demo.json

Fully local when the ESCO and Hugging Face caches are warm:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/export_demo.py --out <path>
"""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tiny_skill_linker.demo import check_top
from tiny_skill_linker.models import PromptedBiEncoder
from tiny_skill_linker.perquery import load_perquery
from tiny_skill_linker.provenance import code_state, weights_digest
from tiny_skill_linker.tasks import ESCO_VERSION, TASKS, load_task

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 1
K = 10
SPLIT = "test"

# The three arms the page compares, each matching a committed results/test row so
# every number on the page is checkable against one. All are fp32 torch with no
# query prefix; a prefix belongs to the baselines, and applying one here would
# silently change the scores.
ARMS = (
    ("stock", "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ("tuned", "models/ft-seed0", "ft-seed0"),
    ("holdout", "models/ft-holdout-seed0", "ft-holdout-seed0"),
)
# The arm whose training excluded a fifth of the skills, and so the one the
# held-out filter is about.
HOLDOUT_ARM = "holdout"
HOLDOUT_RECORD = "ft-holdout-seed0"


def published(label: str) -> dict:
    """The committed aggregate for an arm, so the page can state what it reproduces."""
    record = json.loads((ROOT / "results" / SPLIT / f"{label}.json").read_text())
    return {task: scores["rp@5"] for task, scores in record["scores"].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="where to write the JSON")
    args = ap.parse_args()

    start = time.time()
    tasks = {name: load_task(name, SPLIT) for name in TASKS}
    heldout = set(
        json.loads((ROOT / "results" / "train" / f"{HOLDOUT_RECORD}.json").read_text())[
            "heldout_skills"
        ]
    )

    # A task's own indices are positions in a space WorkRB rebuilds and remaps per
    # version, so the file carries label strings in a table of its own.
    labels: dict[str, int] = {}

    def label_index(name: str) -> int:
        return labels.setdefault(name, len(labels))

    sets: dict[str, dict] = {}
    for name, task in tasks.items():
        dataset = task.datasets["en"]
        sets[name] = {
            "n_queries": len(dataset.query_texts),
            "n_targets": len(dataset.target_space),
            "text": list(dataset.query_texts),
            "gold": [
                [label_index(dataset.target_space[g]) for g in gold]
                for gold in dataset.target_indices
            ],
            "top": {},
        }

    for key, path, label in ARMS:
        model = PromptedBiEncoder(path, query_prompt="", label=label)
        reference = load_perquery(ROOT, SPLIT, label, need_ranks=True)
        for name, task in tasks.items():
            dataset = task.datasets["en"]
            matrix = task.compute_prediction_matrix(model, "en")
            # The descending sort WorkRB scores with, asked to be stable so a rerun
            # of this script repeats itself.
            order = np.argsort(-matrix, axis=1, kind="stable")[:, :K].tolist()
            check_top(dataset.target_indices, order, reference[name])
            space = dataset.target_space
            sets[name]["top"][key] = [[label_index(space[t]) for t in row] for row in order]
        del model

    out = {
        "schema": SCHEMA,
        "split": SPLIT,
        "k": K,
        "esco_version": ESCO_VERSION,
        "arms": {
            key: {"model": path, "label": label, "published_rp5": published(label)}
            for key, path, label in ARMS
        },
        "holdout_arm": HOLDOUT_ARM,
        "labels": list(labels),
        # Positions in `labels` whose skill was kept out of the holdout arm's
        # training, with every sentence that mentions it.
        "heldout": sorted(i for name, i in labels.items() if name in heldout),
        "sets": sets,
        "provenance": {
            "source_repo": "tiny-skill-linker",
            "source_sha": code_state()["sha"],
            "source_dirty": code_state()["dirty"],
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "esco_version": ESCO_VERSION,
            "precision": "fp32",
            "backend": "torch",
            "weights_digest": {
                key: weights_digest(path) for key, path, _ in ARMS if Path(path).exists()
            },
            "versions": {
                p: version(p)
                for p in ("workrb", "sentence-transformers", "transformers", "torch", "datasets")
            },
            "machine": platform.platform(),
            "seconds": round(time.time() - start, 1),
        },
    }

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(dest),
                "bytes": dest.stat().st_size,
                "labels": len(labels),
                "heldout_labels": len(out["heldout"]),
                "queries": sum(s["n_queries"] for s in sets.values()),
                "seconds": out["provenance"]["seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
