"""Deterministic stand-ins for the models, shared by the unit tests."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from arxiv_agent.documents import Chunk, PaperMeta


class HashingEmbeddings(Embeddings):
    """Bag-of-words vectors: texts sharing words are close, unrelated ones are not."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0  # noqa: S324
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def meta(arxiv_id: str = "2210.03629v3", title: str = "ReAct") -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        title=title,
        authors=("A. Author",),
        published="2022-10-06",
        abstract="An abstract.",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def chunk(arxiv_id: str, index: int, text: str, title: str = "Paper", page: int = 1) -> Chunk:
    return Chunk(
        id=f"{arxiv_id}#{index}", arxiv_id=arxiv_id, title=title, page=page, index=index, text=text
    )
