"""Paired-bootstrap comparisons between two evaluated models.

`diff` compares two labels query by query and reports the mean difference with a 95%
bootstrap interval, printing it and writing it to results/<split>/diff/<a>-vs-<b>.json.
Per-query scores are written by scripts/evaluate.py at the end of every run.

Usage:
    python scripts/compare.py diff --split test --a ft-seed0 --b all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tiny_skill_linker.perquery import load_perquery
from tiny_skill_linker.provenance import code_state

ROOT = Path(__file__).resolve().parent.parent
RESAMPLES = 10_000
SEED = 0


def cmd_diff(split: str, a: str, b: str) -> None:
    pa, pb = (load_perquery(ROOT, split, x) for x in (a, b))
    rng = np.random.default_rng(SEED)
    print(f"{a} minus {b} ({split}; paired bootstrap over queries, {RESAMPLES} resamples)")
    out = {}
    for name in pa:
        out[name] = {}
        for metric in ("rp@5", "rr"):
            d = np.array(pa[name][metric]) - np.array(pb[name][metric])
            idx = rng.integers(0, len(d), size=(RESAMPLES, len(d)))
            lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            out[name][metric] = {
                "delta": float(d.mean()),
                "ci95": [float(lo), float(hi)],
                "n": len(d),
            }
            print(
                f"  {name:9s} {metric:5s} {100 * d.mean():+6.2f}  [{100 * lo:+6.2f}, {100 * hi:+6.2f}]"
                f"  n={len(d)}"
            )
    dest = ROOT / "results" / split / "diff" / f"{a}-vs-{b}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "a": a,
                "b": b,
                "split": split,
                "resamples": RESAMPLES,
                "seed": SEED,
                "tasks": out,
                "provenance": {"code": code_state()},
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p2 = sub.add_parser("diff")
    p2.add_argument("--split", required=True)
    p2.add_argument("--a", required=True)
    p2.add_argument("--b", required=True)
    args = ap.parse_args()
    cmd_diff(args.split, args.a, args.b)


if __name__ == "__main__":
    main()
