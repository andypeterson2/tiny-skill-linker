"""Per-query RP@5 and reciprocal rank from WorkRB's saved rankings."""

import json
from pathlib import Path

K = 5
# Schema 1 holds rp@5 and rr only. Schema 2 adds each gold skill's rank, which
# the seen/unseen split needs. A file states which it is.
SCHEMA = 2


def per_query(scores: dict, gold: list[list[int]]) -> dict[str, list]:
    rp, rr, gold_ranks = [], [], []
    for q, positives in enumerate(gold):
        row = scores[str(q)]
        order = sorted(row, key=lambda t: -row[t])
        ranks = {int(t): i for i, t in enumerate(order)}
        hits = sum(1 for p in positives if ranks[p] < K)
        rp.append(hits / min(K, len(positives)))
        rr.append(1.0 / (1 + min(ranks[p] for p in positives)))
        gold_ranks.append([ranks[p] for p in positives])
    return {"rp@5": rp, "rr": rr, "gold": [list(g) for g in gold], "gold_ranks": gold_ranks}


def load_perquery(root: Path, split: str, label: str, need_ranks: bool = False) -> dict:
    """Read results/<split>/perquery/<label>.json.

    With need_ranks, refuse a schema 1 file instead of failing on a missing key.
    """
    path = root / "results" / split / "perquery" / f"{label}.json"
    payload = json.loads(path.read_text())
    if need_ranks and payload["schema"] < 2:
        raise SystemExit(
            f"{path} is schema {payload['schema']} and holds no gold ranks.\n"
            f"Regenerate it: python scripts/evaluate.py --model <model> --split {split} "
            f"--label {label}"
        )
    return payload["tasks"]


def write_perquery(root: Path, split: str, label: str, tasks: dict) -> Path:
    """Reduce runs/<split>/<label>/rankings/ to results/<split>/perquery/<label>.json."""
    ranking = root / "runs" / split / label / "rankings" / label
    out = {}
    for name, task in tasks.items():
        path = ranking / f"{task.name.replace(' ', '_')}__en.json"
        artifact = json.loads(path.read_text())
        out[name] = per_query(artifact["scores"], task.datasets["en"].target_indices)
    dest = root / "results" / split / "perquery" / f"{label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"schema": SCHEMA, "tasks": out}) + "\n")
    return dest
