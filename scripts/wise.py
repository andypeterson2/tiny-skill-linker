"""Interpolate a fine-tuned model's weights with its base model (WiSE-FT).

theta(alpha) = (1 - alpha) * theta_base + alpha * theta_finetuned, applied to every
floating-point tensor; alpha = 0 is the base model and alpha = 1 the fine-tuned one.

Usage:
    python scripts/wise.py --finetuned models/ft-seed1 --alpha 0.5 --out models/wise-seed1-a0.5
"""

import argparse

import torch
from sentence_transformers import SentenceTransformer

BASE = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = SentenceTransformer(args.base, device="cpu").state_dict()
    model = SentenceTransformer(args.finetuned, device="cpu")
    tuned = model.state_dict()
    if base.keys() != tuned.keys():
        raise ValueError("base and fine-tuned models have different parameters")
    blended = {
        k: torch.lerp(base[k], v, args.alpha) if v.is_floating_point() else v
        for k, v in tuned.items()
    }
    model.load_state_dict(blended)
    model.save_pretrained(args.out)
    print(f"alpha={args.alpha} -> {args.out}")


if __name__ == "__main__":
    main()
