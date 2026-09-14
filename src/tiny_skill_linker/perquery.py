"""Per-query RP@5 and reciprocal rank from WorkRB's saved rankings."""

import json
from pathlib import Path

K = 5


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
    dest.write_text(json.dumps(out) + "\n")
    return dest
