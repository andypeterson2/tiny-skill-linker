"""Sentence-transformers bi-encoders exposed through WorkRB's model interface."""

import torch
from sentence_transformers import SentenceTransformer
from workrb.models.base import ModelInterface
from workrb.types import ModelInputType

from tiny_skill_linker.onnx_encoder import OnnxEncoder


class PromptedBiEncoder(ModelInterface):
    """Cosine-similarity ranker with an optional prefix on the query side only."""

    def __init__(
        self,
        model_name_or_path: str,
        query_prompt: str = "",
        label: str | None = None,
        onnx_file: str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.query_prompt = query_prompt
        self.onnx_file = onnx_file
        self._label = label or model_name_or_path.rstrip("/").split("/")[-1]
        if onnx_file:
            self.onnx = OnnxEncoder(model_name_or_path, onnx_file)
        else:
            self.onnx = None
            self.model = SentenceTransformer(model_name_or_path, device="cpu")
            self.model.eval()

    @property
    def name(self) -> str:
        return self._label

    @property
    def description(self) -> str:
        prompt = f" with query prefix {self.query_prompt!r}" if self.query_prompt else ""
        backend = f" ({self.onnx_file})" if self.onnx_file else ""
        return f"{self.model_name_or_path}{backend}{prompt}, cosine similarity"

    @property
    def classification_label_space(self) -> list[str] | None:
        return None

    def encode(self, texts: list[str], prompt: str = "") -> torch.Tensor:
        if self.onnx:
            return self.onnx.encode([prompt + t for t in texts])
        emb = self.model.encode(
            [prompt + t for t in texts],
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        return self.encode(queries, self.query_prompt) @ self.encode(targets).T

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        return self._compute_rankings(texts, targets, input_type, target_input_type)
