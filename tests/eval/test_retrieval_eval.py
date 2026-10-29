from pathlib import Path

import pytest

from arxiv_agent.eval.dataset import EvalItem, Evidence
from arxiv_agent.eval.retrieval_eval import evaluate_retrieval
from arxiv_agent.retrieval.hits import Hit
from arxiv_agent.retrieval.retriever import DenseRetriever
from arxiv_agent.retrieval.store import PaperStore
from tests.fakes import HashingEmbeddings, chunk

TOPICS = [
    "reasoning traces interleaved with environment actions for question answering",
    "verbal reflection stored in episodic memory after failed trials",
    "tree search over candidate thoughts with self evaluation and backtracking",
    "self supervised learning of api calls inserted into text",
    "skill library of executable code for open ended minecraft exploration",
    "graph based community summaries for global sensemaking questions",
]


class Fixed:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query: str, arxiv_id: str | None = None) -> list[Hit]:
        return self.hits


def item(i: int, *evidence: Evidence) -> EvalItem:
    return EvalItem(f"q{i}", f"q{i}", "answer", evidence, "single", False)


def test_mrr_hit_rate_and_recall_from_hand_computed_ranks() -> None:
    quotes = [f"distinctive evidence sentence number {n} for this test" for n in range(3)]
    noise = [Hit(chunk("1000.00000v1", 90 + n, "unrelated filler text"), 0.0) for n in range(4)]

    def at(rank: int, quote: str) -> list[Hit]:
        target = Hit(chunk("1000.00000v1", rank, quote), 1.0)
        return [*noise[: rank - 1], target, *noise[rank - 1 :]]

    report_items = []
    for n, rank in enumerate([1, 3]):
        report_items.append(
            evaluate_retrieval(
                Fixed(at(rank, quotes[n])), [item(n, Evidence("1000.00000v1", quotes[n]))]
            )
        )
    absent = evaluate_retrieval(Fixed(noise), [item(2, Evidence("1000.00000v1", quotes[2]))])
    assert report_items[0].mrr == 1.0
    assert report_items[1].mrr == pytest.approx(1 / 3)
    assert absent.mrr == 0.0
    assert absent.items[0].rank is None

    combined = evaluate_retrieval(
        _Router({"q0": at(1, quotes[0]), "q1": at(3, quotes[1]), "q2": noise}),
        [item(n, Evidence("1000.00000v1", quotes[n])) for n in range(3)],
    )
    assert combined.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert combined.hit_rate == pytest.approx(2 / 3)
    assert combined.recall == pytest.approx(2 / 3)


class _Router:
    """Returns a fixed ranking per question id (the question text carries the id)."""

    def __init__(self, by_id: dict[str, list[Hit]]) -> None:
        self.by_id = by_id

    def search(self, query: str, arxiv_id: str | None = None) -> list[Hit]:
        return self.by_id[query]


def test_the_harness_ranks_a_reversed_retriever_below_the_real_one(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "c", HashingEmbeddings(), "eval")
    for p, topic in enumerate(TOPICS):
        store.replace_paper(
            [chunk(f"100{p}.00000v1", 0, topic)]
            + [chunk(f"100{p}.00000v1", i, f"filler words {p} {i}") for i in range(1, 4)]
        )
    items = [
        EvalItem(f"q{p}", topic, "a", (Evidence(f"100{p}.00000v1", topic),), "single", False)
        for p, topic in enumerate(TOPICS)
    ]

    class Reversed:
        def search(self, query: str, arxiv_id: str | None = None) -> list[Hit]:
            return list(reversed(store.dense_search(query, 20)))[:5]

    real = evaluate_retrieval(DenseRetriever(store, k=5), items)
    assert real.mrr == 1.0
    assert evaluate_retrieval(Reversed(), items).mrr < real.mrr
