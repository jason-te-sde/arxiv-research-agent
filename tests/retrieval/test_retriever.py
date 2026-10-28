import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from arxiv_agent.retrieval.retriever import DenseRetriever, TwoStageRetriever
from arxiv_agent.retrieval.store import PaperStore
from tests.fakes import HashingEmbeddings, chunk


class ListReranker:
    """Scores candidates by a fixed function of their position."""

    def __init__(self, by_position: list[float] | None = None) -> None:
        self.by_position = by_position
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if self.by_position is None:
            return [float(i) for i in range(len(texts))]  # later candidates score higher
        return self.by_position[: len(texts)]


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    s = PaperStore(tmp_path / "chroma", HashingEmbeddings(), "test")
    for p in range(3):
        arxiv_id = f"100{p}.00000v1"
        s.replace_paper(
            [chunk(arxiv_id, i, f"shared topic word{p}{i} filler{i}") for i in range(10)]
        )
    return s


def test_dense_retriever_returns_k_hits(store: PaperStore) -> None:
    assert len(DenseRetriever(store, k=5).search("shared topic")) == 5


def test_reranker_order_decides_the_final_top_k(store: PaperStore) -> None:
    dense = store.dense_search("shared topic", k=20)
    hits = TwoStageRetriever(store, ListReranker(), k_candidates=20, k=5).search("shared topic")
    assert [h.chunk.id for h in hits] == [h.chunk.id for h in reversed(dense[-5:])]
    assert [h.score for h in hits] == [19.0, 18.0, 17.0, 16.0, 15.0]


def test_ties_keep_dense_order(store: PaperStore) -> None:
    dense = store.dense_search("shared topic", k=20)
    flat = ListReranker([0.0] * 20)
    hits = TwoStageRetriever(store, flat, k_candidates=20, k=5).search("shared topic")
    assert [h.chunk.id for h in hits] == [h.chunk.id for h in dense[:5]]


def test_reranker_sees_the_title_and_body(store: PaperStore) -> None:
    reranker = ListReranker()
    TwoStageRetriever(store, reranker, k_candidates=3, k=2).search("shared topic")
    query, texts = reranker.calls[0]
    assert query == "shared topic"
    assert len(texts) == 3
    assert all(t.startswith("Paper\n") for t in texts)


def test_fewer_candidates_than_k_returns_them_all(tmp_path: Path) -> None:
    s = PaperStore(tmp_path / "c", HashingEmbeddings(), "tiny")
    s.replace_paper([chunk("1000.00000v1", 0, "lonely chunk")])
    hits = TwoStageRetriever(s, ListReranker(), k_candidates=20, k=5).search("lonely")
    assert len(hits) == 1


def test_paper_filter_reaches_the_dense_stage(store: PaperStore) -> None:
    hits = TwoStageRetriever(store, ListReranker(), k_candidates=20, k=5).search(
        "shared topic", arxiv_id="1001.00000v1"
    )
    assert {h.chunk.arxiv_id for h in hits} == {"1001.00000v1"}


def test_empty_store_reranks_nothing(tmp_path: Path) -> None:
    reranker = ListReranker()
    s = PaperStore(tmp_path / "c", HashingEmbeddings(), "tiny")
    assert TwoStageRetriever(s, reranker).search("anything") == []
    assert reranker.calls == []


VOCAB = (
    "agent",
    "tool",
    "plan",
    "memory",
    "reward",
    "search",
    "retrieval",
    "reasoning",
    "action",
    "observation",
    "language",
    "model",
    "policy",
    "environment",
    "benchmark",
    "web",
    "code",
    "program",
    "feedback",
    "reflection",
    "critique",
    "summary",
    "graph",
    "tree",
    "node",
    "value",
    "step",
    "trajectory",
    "prompt",
    "example",
    "demonstration",
    "task",
    "question",
    "answer",
    "document",
    "passage",
    "query",
    "index",
    "vector",
    "embedding",
    "score",
    "rank",
    "error",
)


@pytest.mark.slow
def test_chroma_search_equals_brute_force(tmp_path: Path) -> None:
    from arxiv_agent.retrieval.embeddings import make_embeddings

    rng = random.Random(0)  # noqa: S311 - seeded test data, not secrets
    embeddings = make_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    store = PaperStore(tmp_path / "chroma", embeddings, "diff")
    for p in range(10):
        store.replace_paper(
            [chunk(f"20{p:02d}.00000v1", i, " ".join(rng.choices(VOCAB, k=30))) for i in range(30)]
        )
    chunks, vectors = store.all_embeddings()
    for _ in range(20):
        query = " ".join(rng.choices(VOCAB, k=8))
        q = np.asarray(embeddings.embed_query(query), dtype=np.float32)
        sims = vectors @ q
        exact = np.argsort(-sims, kind="stable")[:20]
        hits = store.dense_search(query, k=20)
        # Same 20 chunks, same similarity at every rank. Ids are not compared rank by
        # rank: candidates 1e-7 apart (float32 noise) may legitimately swap places.
        assert {h.chunk.id for h in hits} == {chunks[i].id for i in exact}
        assert [h.score for h in hits] == pytest.approx([float(sims[i]) for i in exact], abs=1e-5)


@pytest.mark.slow
def test_cross_encoder_prefers_the_relevant_passage() -> None:
    from arxiv_agent.retrieval.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    on, off = reranker.score(
        "How does ReAct combine reasoning and acting?",
        [
            "ReAct interleaves reasoning traces with task-specific actions.",
            "The bakery opens at seven and sells sourdough bread.",
        ],
    )
    assert on > off
