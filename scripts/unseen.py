"""Split test results by whether each gold skill was held out of training.

Every (sentence, gold skill) pair on the test sets is labelled "seen" or "unseen"
using the held-out list a training run recorded, then each model's hit@5 and
reciprocal rank are reported per group, with a paired bootstrap over pairs for
the difference between two models.

Usage:
    python scripts/unseen.py --holdout ft-holdout-seed0 \
        --models all-MiniLM-L6-v2 ft-seed0 ft-holdout-seed0 --a ft-holdout-seed0 --b all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tiny_skill_linker.tasks import TASKS, load_task

ROOT = Path(__file__).resolve().parent.parent
K = 5
RESAMPLES = 10_000


def pairs(label: str, heldout: set[str], spaces: dict[str, list[str]]) -> list[dict]:
    perquery = json.loads((ROOT / "results" / "test" / "perquery" / f"{label}.json").read_text())
    out = []
    for task, scores in perquery.items():
        for q, (gold, ranks) in enumerate(zip(scores["gold"], scores["gold_ranks"], strict=True)):
            for g, r in zip(gold, ranks, strict=True):
                skill = spaces[task][g]
                out.append(
                    {"task": task, "q": q, "skill": skill, "unseen": skill in heldout, "rank": r}
                )
    return out


def summarize(rows: list[dict]) -> dict:
    hit = np.array([r["rank"] < K for r in rows], dtype=float)
    rr = np.array([1.0 / (1 + r["rank"]) for r in rows])
    return {"n": len(rows), "hit@5": float(hit.mean()), "mrr": float(rr.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--holdout", required=True, help="training run whose held-out skills define 'unseen'"
    )
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    record = json.loads((ROOT / "results" / "train" / f"{args.holdout}.json").read_text())
    heldout = set(record["heldout_skills"])
    spaces = {n: load_task(n, "test").datasets["en"].target_space for n in TASKS}
    vocabulary = set(spaces["tech"])
    print(
        f"held-out skills: {len(heldout)}; of these in the ESCO test vocabulary: {len(heldout & vocabulary)}"
    )

    table = {}
    for label in args.models:
        rows = pairs(label, heldout, spaces)
        table[label] = {
            group: summarize([r for r in rows if r["unseen"] == unseen])
            for group, unseen in (("seen", False), ("unseen", True))
        }
        s, u = table[label]["seen"], table[label]["unseen"]
        print(
            f"{label:22s} seen: hit@5 {100 * s['hit@5']:5.2f}  MRR {100 * s['mrr']:5.2f}  (n={s['n']})"
            f" | unseen: hit@5 {100 * u['hit@5']:5.2f}  MRR {100 * u['mrr']:5.2f}  (n={u['n']})"
        )

    rng = np.random.default_rng(0)
    ra, rb = pairs(args.a, heldout, spaces), pairs(args.b, heldout, spaces)
    diffs = {}
    for group, unseen in (("seen", False), ("unseen", True)):
        a = np.array([r["rank"] < K for r in ra if r["unseen"] == unseen], dtype=float)
        b = np.array([r["rank"] < K for r in rb if r["unseen"] == unseen], dtype=float)
        d = a - b
        idx = rng.integers(0, len(d), size=(RESAMPLES, len(d)))
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        diffs[group] = {"delta_hit@5": float(d.mean()), "ci95": [float(lo), float(hi)], "n": len(d)}
        print(
            f"{args.a} minus {args.b}, {group}: hit@5 {100 * d.mean():+.2f}"
            f"  [{100 * lo:+.2f}, {100 * hi:+.2f}]  n={len(d)}"
        )

    dest = ROOT / "results" / "test" / f"unseen-{args.holdout}.json"
    dest.write_text(
        json.dumps(
            {
                "holdout": args.holdout,
                "k": K,
                "models": table,
                "diff": {"a": args.a, "b": args.b, **diffs},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
