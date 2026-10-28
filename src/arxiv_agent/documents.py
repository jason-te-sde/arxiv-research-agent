"""The two records every layer shares: a paper's metadata and one of its chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str  # always versioned, e.g. "2210.03629v3"
    title: str
    authors: tuple[str, ...]
    published: str  # ISO date
    abstract: str
    pdf_url: str


@dataclass(frozen=True)
class Chunk:
    id: str  # "{arxiv_id}#{index}"; answers cite this id in square brackets
    arxiv_id: str
    title: str
    page: int  # 1-based page where the chunk starts
    index: int
    text: str

    @property
    def embed_text(self) -> str:
        """What the embedder sees: the title gives every chunk its paper's context."""
        return f"{self.title}\n{self.text}"
