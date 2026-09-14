"""Lay out an exported model the way transformers.js loads it with dtype "q8".

transformers.js reads config.json and the tokenizer files from the model root and
the int8 weights from onnx/model_quantized.onnx.

Usage:
    python scripts/package_tjs.py --model models/ft-seed1 --export models/ft-seed1-onnx \
        --out models/tjs/tiny-skill-linker
"""

import argparse
import shutil
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="sentence-transformers model directory")
    ap.add_argument("--export", required=True, help="scripts/export_int8.py output directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, export, out = Path(args.model), Path(args.export), Path(args.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    shutil.copy2(model / "config.json", out / "config.json")
    for name in TOKENIZER_FILES:
        if (export / name).exists():
            shutil.copy2(export / name, out / name)
    shutil.copy2(export / "onnx" / "model_int8.onnx", out / "onnx" / "model_quantized.onnx")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"{f.relative_to(out)}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
