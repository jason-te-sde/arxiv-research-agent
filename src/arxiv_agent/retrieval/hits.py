from __future__ import annotations

from dataclasses import dataclass

from arxiv_agent.documents import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity for dense hits, cross-encoder logit once reranked
