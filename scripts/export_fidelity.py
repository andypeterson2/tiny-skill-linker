"""Compare two encoders' embeddings of the same ESCO skill names.

This is the check behind the choice of quantization settings: an export whose embeddings
track the deployed q8 file closely is the one to ship. It reads skill names from the task
target space, never from the query side, so no test sentence is involved.

Usage:
    python scripts/export_fidelity.py --a models/minilm --a-onnx onnx/model_int8.onnx \
        --b models/xenova --b-onnx onnx/model_quantized.onnx --label int8-vs-xenova-q8
"""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import torch

from tiny_skill_linker.onnx_encoder import OnnxEncoder
from tiny_skill_linker.provenance import code_state, weights_digest
from tiny_skill_linker.tasks import ESCO_VERSION, load_task

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = 2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-onnx", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-onnx", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sample", type=int, default=SAMPLE)
    args = ap.parse_args()

    space = sorted(load_task("tech", "test").datasets["en"].target_space)
    # Evenly spaced over the sorted vocabulary, so the sample is fixed without a seed.
    step = max(1, len(space) // args.sample)
    names = space[::step][: args.sample]

    start = time.time()
    ea = OnnxEncoder(args.a, args.a_onnx).encode(names)
    eb = OnnxEncoder(args.b, args.b_onnx).encode(names)
    cosine = torch.nn.functional.cosine_similarity(ea, eb, dim=1)

    out = {
        "label": args.label,
        "a": {"model": args.a, "onnx_file": args.a_onnx},
        "b": {"model": args.b, "onnx_file": args.b_onnx},
        "n_names": len(names),
        "cosine": {
            "mean": float(cosine.mean()),
            "min": float(cosine.min()),
            "p05": float(cosine.quantile(0.05)),
        },
        "provenance": {
            "esco_version": ESCO_VERSION,
            "source": "ESCO skill names from the tech test target space",
            "a_weights_digest": weights_digest(args.a, args.a_onnx),
            "b_weights_digest": weights_digest(args.b, args.b_onnx),
            "code": code_state(),
            "versions": {p: version(p) for p in ("onnxruntime", "transformers", "torch")},
            "machine": platform.platform(),
            "seconds": round(time.time() - start, 1),
        },
    }
    dest = ROOT / "results" / "fidelity" / f"{args.label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{args.label}: mean cosine {out['cosine']['mean']:.4f} over {len(names)} names")


if __name__ == "__main__":
    main()
