"""Export a sentence-transformers model's encoder to ONNX, plus a dynamically quantized int8 copy.

The output directory holds tokenizer files, onnx/model.onnx, onnx/model_int8.onnx and
pooling.json, which is everything OnnxEncoder needs to embed text without torch.

Usage:
    python scripts/export_int8.py --model sentence-transformers/all-MiniLM-L6-v2 --out models/minilm
"""

import argparse
import json
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sentence_transformers import SentenceTransformer

OPSET = 14


def pooling_mode(model: SentenceTransformer) -> str:
    pooling = next(m for m in model if type(m).__name__ == "Pooling")
    mode = pooling.get_config_dict()["pooling_mode"]
    if mode not in ("cls", "mean"):
        raise ValueError(f"unsupported pooling mode {mode!r}; OnnxEncoder handles cls and mean")
    return mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(args.model, device="cpu")
    encoder = st[0].auto_model.eval()
    st.tokenizer.save_pretrained(str(out))

    sample = st.tokenizer(["a sample sentence"], return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    axes = {n: {0: "batch", 1: "sequence"} for n in names}
    axes["last_hidden_state"] = {0: "batch", 1: "sequence"}

    fp32 = out / "onnx" / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            tuple(sample[n] for n in names),
            str(fp32),
            input_names=names,
            output_names=["last_hidden_state"],
            dynamic_axes=axes,
            opset_version=OPSET,
        )
    int8 = out / "onnx" / "model_int8.onnx"
    # Per-channel with reduced range: the setting whose embeddings match the deployed q8 model.
    quantize_dynamic(
        str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True
    )

    meta = {
        "source": args.model,
        "pooling": pooling_mode(st),
        "normalize": any(type(m).__name__ == "Normalize" for m in st),
        "max_seq_length": st.max_seq_length,
    }
    (out / "pooling.json").write_text(json.dumps(meta, indent=2) + "\n")
    for f in (fp32, int8):
        print(f"{f.relative_to(out)}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
