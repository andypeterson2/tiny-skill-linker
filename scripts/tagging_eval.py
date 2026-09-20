"""Score a bi-encoder on a small tagging set: rank a tag vocabulary for each text.

This is the shape of check used for coarse category tags, as opposed to fine-grained skill
linking. It takes any JSONL of items and any tag list, so a private set stays outside this
repository: only the metrics and the input file's SHA-256 are written.

Each line of --items is {"text": str, "gold_tags": [str], "cluster": str (optional)}.
The bootstrap resamples clusters, so several items from one document count once.

Usage:
    python scripts/tagging_eval.py --items private/bullets.jsonl --tags private/tags.txt \
        --model models/tjs/tiny-skill-linker --onnx-file onnx/model_quantized.onnx \
        --label ft-seed1-int8
    python scripts/tagging_eval.py ... --against results/tagging/stock.json
"""

import argparse
import json
import platform
import time
from math import comb
from pathlib import Path

import numpy as np

from tiny_skill_linker.models import PromptedBiEncoder
from tiny_skill_linker.provenance import code_state, file_digest, weights_digest

ROOT = Path(__file__).resolve().parent.parent
RESAMPLES = 10_000
SEED = 0
KS = (1, 3)


def ranks_of_gold(items: list[dict], tags: list[str], model: PromptedBiEncoder) -> list[int]:
    """Rank of each item's best gold tag, zero-indexed."""
    index = {t: i for i, t in enumerate(tags)}
    scores = model.encode([i["text"] for i in items], model.query_prompt) @ model.encode(tags).T
    order = scores.argsort(dim=1, descending=True).tolist()
    out = []
    for item, row in zip(items, order, strict=True):
        position = {t: p for p, t in enumerate(row)}
        out.append(min(position[index[g]] for g in item["gold_tags"]))
    return out


def summarize(ranks: list[int]) -> dict:
    ranks = np.array(ranks)
    out = {f"hit@{k}": float((ranks < k).mean()) for k in KS}
    out["mrr"] = float((1.0 / (1 + ranks)).mean())
    return out


def cluster_bootstrap(a: np.ndarray, b: np.ndarray, clusters: list[str]) -> dict:
    """95% interval for mean(a) - mean(b), resampling whole clusters."""
    groups = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(RESAMPLES):
        picked = rng.integers(0, len(keys), size=len(keys))
        idx = [i for p in picked for i in groups[keys[p]]]
        means.append((a[idx] - b[idx]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": float((a - b).mean()), "ci95": [float(lo), float(hi)], "n_clusters": len(keys)}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided sign test over the items where the two models disagree."""
    wins = int(((a == 1) & (b == 0)).sum())
    losses = int(((a == 0) & (b == 1)).sum())
    n = wins + losses
    if n == 0:
        return {"better": wins, "worse": losses, "p": 1.0}
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return {"better": wins, "worse": losses, "p": min(1.0, 2 * tail / 2**n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="JSONL of {text, gold_tags, cluster?}")
    ap.add_argument("--tags", required=True, help="one tag per line")
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx-file", default=None)
    ap.add_argument("--query-prompt", default="")
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--label", required=True)
    ap.add_argument("--against", default=None, help="an earlier results file to compare with")
    args = ap.parse_args()

    items_path, tags_path = Path(args.items), Path(args.tags)
    items = [json.loads(line) for line in items_path.read_text().splitlines() if line.strip()]
    tags = [t for t in tags_path.read_text().splitlines() if t.strip()]
    clusters = [i.get("cluster", str(n)) for n, i in enumerate(items)]
    unknown = {g for i in items for g in i["gold_tags"]} - set(tags)
    if unknown:
        raise SystemExit(f"gold tags missing from the tag list: {sorted(unknown)}")

    start = time.time()
    model = PromptedBiEncoder(
        args.model, query_prompt=args.query_prompt, label=args.label, onnx_file=args.onnx_file
    )
    ranks = ranks_of_gold(items, tags, model)

    out = {
        "label": args.label,
        "model": args.model,
        "precision": args.precision,
        "backend": f"onnx:{args.onnx_file}" if args.onnx_file else "torch",
        "n_items": len(items),
        "n_tags": len(tags),
        "scores": summarize(ranks),
        "ranks": ranks,
        "clusters": clusters,
        "provenance": {
            "items_sha256": file_digest(items_path),
            "tags_sha256": file_digest(tags_path),
            "weights_digest": weights_digest(args.model, args.onnx_file),
            "code": code_state(),
            "machine": platform.platform(),
            "seconds": round(time.time() - start, 1),
        },
    }

    if args.against:
        other = json.loads(Path(args.against).read_text())
        if other["provenance"]["items_sha256"] != out["provenance"]["items_sha256"]:
            raise SystemExit("the two runs scored different item files")
        mine, theirs = np.array(ranks), np.array(other["ranks"])
        out["diff"] = {"against": other["label"]}
        for k in KS:
            a, b = (mine < k).astype(float), (theirs < k).astype(float)
            out["diff"][f"hit@{k}"] = {**cluster_bootstrap(a, b, clusters), **mcnemar(a, b)}
        a, b = 1.0 / (1 + mine), 1.0 / (1 + theirs)
        out["diff"]["mrr"] = cluster_bootstrap(a, b, clusters)

    dest = ROOT / "results" / "tagging" / f"{args.label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"scores": out["scores"], "diff": out.get("diff")}, indent=2))


if __name__ == "__main__":
    main()
