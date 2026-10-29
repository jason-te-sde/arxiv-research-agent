"""Retrieval quality over a question set: MRR@k, hit rate@k and evidence recall@k."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arxiv_agent.eval.dataset import EvalItem
from arxiv_agent.eval.relevance import first_relevant_rank, recall_at_k
from arxiv_agent.retrieval.retriever import Retriever


@dataclass(frozen=True)
class ItemResult:
    id: str
    rank: int | None  # 1-based rank of the first relevant chunk within k, or None
    recall: float
    retrieved: tuple[str, ...]  # chunk ids, in rank order


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    mrr: float
    hit_rate: float
    recall: float
    items: tuple[ItemResult, ...]


def evaluate_retrieval(
    retriever: Retriever, items: Sequence[EvalItem], k: int = 5
) -> RetrievalReport:
    results = []
    for item in items:
        hits = retriever.search(item.question)[:k]
        results.append(
            ItemResult(
                id=item.id,
                rank=first_relevant_rank(hits, item.evidence, k),
                recall=recall_at_k(hits, item.evidence, k),
                retrieved=tuple(h.chunk.id for h in hits),
            )
        )
    n = len(results) or 1
    return RetrievalReport(
        k=k,
        mrr=sum(1.0 / r.rank for r in results if r.rank) / n,
        hit_rate=sum(1 for r in results if r.rank) / n,
        recall=sum(r.recall for r in results) / n,
        items=tuple(results),
    )
