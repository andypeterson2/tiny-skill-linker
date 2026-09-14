"""Per-query scores and paired-bootstrap comparisons between two evaluated models.

`perquery` reduces WorkRB's saved rankings (runs/<split>/<label>/rankings/) to per-query RP@5
and reciprocal rank in results/<split>/perquery/<label>.json. `diff` compares two labels query
by query and reports the mean difference with a 95% bootstrap interval.

Usage:
    python scripts/compare.py perquery --split test --label all-MiniLM-L6-v2
    python scripts/compare.py diff --split test --a ft-seed0 --b all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tiny_skill_linker.tasks import TASKS, VAL_TASKS, load_task

ROOT = Path(__file__).resolve().parent.parent
K = 5
RESAMPLES = 10_000


def per_query(scores: dict, gold: list[list[int]]) -> dict[str, list[float]]:
    rp, rr = [], []
    for q, positives in enumerate(gold):
        row = scores[str(q)]
        order = sorted(row, key=lambda t: -row[t])
        ranks = {int(t): i for i, t in enumerate(order)}
        hits = sum(1 for p in positives if ranks[p] < K)
        rp.append(hits / min(K, len(positives)))
        rr.append(1.0 / (1 + min(ranks[p] for p in positives)))
    return {"rp@5": rp, "rr": rr}


def cmd_perquery(split: str, label: str) -> None:
    names = [n for n in TASKS if split == "test" or n in VAL_TASKS]
    out = {}
    for name in names:
        task = load_task(name, split)
        ranking = ROOT / "runs" / split / label / "rankings" / label
        path = ranking / f"{task.name.replace(' ', '_')}__en.json"
        artifact = json.loads(path.read_text())
        out[name] = per_query(artifact["scores"], task.datasets["en"].target_indices)
        print(f"{name}: RP@5 {np.mean(out[name]['rp@5']):.4f}  MRR {np.mean(out[name]['rr']):.4f}")
    dest = ROOT / "results" / split / "perquery" / f"{label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out) + "\n")


def cmd_diff(split: str, a: str, b: str, seed: int = 0) -> None:
    base = ROOT / "results" / split / "perquery"
    pa, pb = (json.loads((base / f"{x}.json").read_text()) for x in (a, b))
    rng = np.random.default_rng(seed)
    print(f"{a} minus {b} ({split}; paired bootstrap over queries, {RESAMPLES} resamples)")
    for name in pa:
        for metric in ("rp@5", "rr"):
            d = np.array(pa[name][metric]) - np.array(pb[name][metric])
            idx = rng.integers(0, len(d), size=(RESAMPLES, len(d)))
            lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            print(
                f"  {name:9s} {metric:5s} {100 * d.mean():+6.2f}  [{100 * lo:+6.2f}, {100 * hi:+6.2f}]"
                f"  n={len(d)}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("perquery")
    p1.add_argument("--split", required=True)
    p1.add_argument("--label", required=True)
    p2 = sub.add_parser("diff")
    p2.add_argument("--split", required=True)
    p2.add_argument("--a", required=True)
    p2.add_argument("--b", required=True)
    args = ap.parse_args()
    if args.cmd == "perquery":
        cmd_perquery(args.split, args.label)
    else:
        cmd_diff(args.split, args.a, args.b)


if __name__ == "__main__":
    main()
