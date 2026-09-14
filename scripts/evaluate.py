"""Score one embedder on TechWolf's ESCO skill-linking sets and write a results JSON.

Usage:
    python scripts/evaluate.py --model sentence-transformers/all-mpnet-base-v2 --split test
    python scripts/evaluate.py --model Snowflake/snowflake-arctic-embed-xs \
        --query-prompt "Represent this sentence for searching relevant passages: " --split val
"""

import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi
from workrb import evaluate

from tiny_skill_linker.models import PromptedBiEncoder
from tiny_skill_linker.tasks import ESCO_VERSION, TASKS, VAL_TASKS, load_task

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["rp@5", "rp@10", "mrr", "map"]


def git_state() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return {"sha": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def hub_revision(repo_id: str, repo_type: str) -> str | None:
    if Path(repo_id).exists():
        return None
    info = HfApi().repo_info(repo_id, repo_type=repo_type)
    return info.sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--query-prompt", default="")
    ap.add_argument("--label", default=None)
    ap.add_argument("--split", choices=["test", "val"], required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--onnx-file", default=None, help="e.g. onnx/model_int8.onnx")
    args = ap.parse_args()

    names = [t for t in args.tasks if args.split == "test" or t in VAL_TASKS]
    tasks = [load_task(n, args.split) for n in names]
    model = PromptedBiEncoder(
        args.model, query_prompt=args.query_prompt, label=args.label, onnx_file=args.onnx_file
    )

    run_dir = ROOT / "runs" / args.split / model.name
    start = time.time()
    results = evaluate(
        model=model,
        tasks=tasks,
        output_folder=str(run_dir),
        metrics={t.name: METRICS for t in tasks},
        force_restart=True,
        save_rankings=True,
    )
    elapsed = time.time() - start

    scores = {}
    for name, task in zip(names, tasks, strict=True):
        result = results.task_results[task.name]
        per_dataset = result.datasetid_results["en"]
        scores[name] = {
            "n_queries": len(task.datasets["en"].query_texts),
            "n_targets": len(task.datasets["en"].target_space),
            **{m: per_dataset.metrics_dict[m] for m in METRICS},
        }

    out = {
        "model": args.model,
        "label": model.name,
        "query_prompt": args.query_prompt,
        "split": args.split,
        "precision": "int8"
        if args.onnx_file and ("int8" in args.onnx_file or "quantized" in args.onnx_file)
        else "fp32",
        "backend": f"onnx:{args.onnx_file}" if args.onnx_file else "torch",
        "scores": scores,
        "provenance": {
            "esco_version": ESCO_VERSION,
            "model_revision": hub_revision(args.model, "model"),
            "dataset_revisions": {n: hub_revision(TASKS[n][1], "dataset") for n in names},
            "code": git_state(),
            "versions": {
                p: version(p)
                for p in ("workrb", "sentence-transformers", "transformers", "torch", "datasets")
            },
            "machine": platform.platform(),
            "seconds": round(elapsed, 1),
        },
    }
    dest = ROOT / "results" / args.split / f"{model.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps({n: {m: round(v, 4) for m, v in s.items()} for n, s in scores.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
