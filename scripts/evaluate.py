"""Score one embedder on TechWolf's ESCO skill-linking sets and write a results JSON.

Usage:
    python scripts/evaluate.py --model sentence-transformers/all-mpnet-base-v2 --split test
    python scripts/evaluate.py --model Snowflake/snowflake-arctic-embed-xs \
        --query-prompt "Represent this sentence for searching relevant passages: " --split val
    python scripts/evaluate.py --model models/minilm --onnx-file onnx/model_int8.onnx \
        --precision int8 --label all-MiniLM-L6-v2-int8 --split test
"""

import argparse
import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi
from workrb import evaluate

from tiny_skill_linker.models import PromptedBiEncoder
from tiny_skill_linker.perquery import write_perquery
from tiny_skill_linker.provenance import code_state, weights_digest
from tiny_skill_linker.tasks import ESCO_VERSION, TASKS, VAL_TASKS, load_task

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["rp@5", "rp@10", "mrr", "map"]


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
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--onnx-file", default=None, help="e.g. onnx/model_int8.onnx")
    ap.add_argument("--note", default=None, help="free text kept with the result")
    args = ap.parse_args()

    names = [t for t in TASKS if args.split == "test" or t in VAL_TASKS]
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
        "precision": args.precision,
        "backend": f"onnx:{args.onnx_file}" if args.onnx_file else "torch",
        "note": args.note,
        "scores": scores,
        "provenance": {
            "esco_version": ESCO_VERSION,
            "model_revision": hub_revision(args.model, "model"),
            "weights_digest": weights_digest(args.model, args.onnx_file),
            "dataset_revisions": {
                n: hub_revision(t.hf_name, "dataset") for n, t in zip(names, tasks, strict=True)
            },
            "code": code_state(),
            "versions": {
                p: version(p)
                for p in ("workrb", "sentence-transformers", "transformers", "torch", "datasets")
            },
            "machine": platform.platform(),
            "seconds": round(elapsed, 1),
        },
    }
    # Full rankings run to ~150 MB per task; keep only the per-query scores.
    write_perquery(ROOT, args.split, model.name, dict(zip(names, tasks, strict=True)))
    shutil.rmtree(run_dir / "rankings")

    dest = ROOT / "results" / args.split / f"{model.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps({n: {m: round(v, 4) for m, v in s.items()} for n, s in scores.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
