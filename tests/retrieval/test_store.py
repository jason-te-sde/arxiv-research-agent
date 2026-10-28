from pathlib import Path

import pytest

from arxiv_agent.retrieval.store import PaperStore, collection_name
from tests.fakes import HashingEmbeddings, chunk


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    return PaperStore(tmp_path / "chroma", HashingEmbeddings(), "test")


def seed(store: PaperStore) -> None:
    store.replace_paper(
        [
            chunk("1111.11111v1", 0, "alpha beta gamma", title="First"),
            chunk("1111.11111v1", 1, "delta epsilon", title="First"),
        ]
    )
    store.replace_paper([chunk("2222.22222v1", 0, "alpha zeta eta", title="Second")])


def test_empty_store_returns_nothing(store: PaperStore) -> None:
    assert store.dense_search("alpha", k=5) == []
    assert store.papers() == []


def test_dense_search_ranks_by_similarity(store: PaperStore) -> None:
    seed(store)
    hits = store.dense_search("delta epsilon", k=3)
    assert hits[0].chunk.id == "1111.11111v1#1"
    assert hits[0].score == pytest.approx(max(h.score for h in hits))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_round_trips_the_chunk(store: PaperStore) -> None:
    seed(store)
    hit = store.dense_search("delta epsilon", k=1)[0]
    assert hit.chunk == chunk("1111.11111v1", 1, "delta epsilon", title="First")


def test_filter_restricts_to_one_paper(store: PaperStore) -> None:
    seed(store)
    hits = store.dense_search("alpha", k=5, arxiv_id="2222.22222v1")
    assert {h.chunk.arxiv_id for h in hits} == {"2222.22222v1"}


def test_replacing_a_paper_does_not_duplicate_it(store: PaperStore) -> None:
    seed(store)
    store.replace_paper([chunk("1111.11111v1", 0, "only one chunk now", title="First")])
    rows = {p.arxiv_id: p.chunks for p in store.papers()}
    assert rows == {"1111.11111v1": 1, "2222.22222v1": 1}


def test_chunks_for_returns_index_order(store: PaperStore) -> None:
    seed(store)
    assert [c.index for c in store.chunks_for("1111.11111v1")] == [0, 1]
    assert store.chunks_for("9999.99999v1") == []


def test_has_paper(store: PaperStore) -> None:
    seed(store)
    assert store.has_paper("2222.22222v1")
    assert not store.has_paper("9999.99999v1")


def test_all_embeddings_aligns_chunks_and_vectors(store: PaperStore) -> None:
    seed(store)
    chunks, vectors = store.all_embeddings()
    assert vectors.shape == (3, 256)
    assert len(chunks) == 3


def test_store_persists_across_instances(tmp_path: Path) -> None:
    seed(PaperStore(tmp_path / "chroma", HashingEmbeddings(), "test"))
    reopened = PaperStore(tmp_path / "chroma", HashingEmbeddings(), "test")
    assert len(reopened.papers()) == 2


def test_collection_name_encodes_model_and_chunking() -> None:
    name = collection_name("sentence-transformers/all-MiniLM-L6-v2", 256, 32)
    assert name == "papers-all-minilm-l6-v2-256-32"
