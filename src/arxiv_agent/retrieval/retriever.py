"""Dense retrieval, and the two-stage pipeline that reranks its candidates."""

from __future__ import annotations

from typing import Protocol

from arxiv_agent.retrieval.hits import Hit
from arxiv_agent.retrieval.rerank import Reranker
from arxiv_agent.retrieval.store import PaperStore


class Retriever(Protocol):
    def search(self, query: str, arxiv_id: str | None = None) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, store: PaperStore, k: int = 5) -> None:
        self._store = store
        self._k = k

    def search(self, query: str, arxiv_id: str | None = None) -> list[Hit]:
        return self._store.dense_search(query, self._k, arxiv_id)


class TwoStageRetriever:
    """Embedding search for ``k_candidates``, then a cross-encoder picks the top ``k``."""

    def __init__(
        self, store: PaperStore, reranker: Reranker, k_candidates: int = 20, k: int = 5
    ) -> None:
        self._store = store
        self._reranker = reranker
        self._k_candidates = k_candidates
        self._k = k

    def search(self, query: str, arxiv_id: str | None = None) -> list[Hit]:
        candidates = self._store.dense_search(query, self._k_candidates, arxiv_id)
        if not candidates:
            return []
        scores = self._reranker.score(query, [h.chunk.embed_text for h in candidates])
        # sorted() is stable, so equal scores keep the dense order.
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [Hit(candidates[i].chunk, scores[i]) for i in order[: self._k]]
