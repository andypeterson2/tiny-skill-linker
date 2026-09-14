"""Embed text with an exported ONNX encoder the way transformers.js's feature-extraction does."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer


class OnnxEncoder:
    def __init__(
        self,
        model_dir: str,
        onnx_file: str,
        pooling: str | None = None,
        max_seq_length: int | None = None,
    ):
        root = Path(model_dir)
        meta_path = root / "pooling.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.pooling = pooling or meta.get("pooling", "mean")
        self.max_seq_length = max_seq_length or meta.get("max_seq_length", 256)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(root / onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in batch.items() if k in self.input_names}
            if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            hidden = self.session.run(None, feed)[0]
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = batch["attention_mask"][..., None].astype(np.float32)
                pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            chunks.append(pooled)
        emb = torch.from_numpy(np.concatenate(chunks))
        return torch.nn.functional.normalize(emb, p=2, dim=1)
