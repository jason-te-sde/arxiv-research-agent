from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly; slower than embeddings, sharper ranking."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device="cpu")

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]
