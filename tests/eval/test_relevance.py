import pytest

from arxiv_agent.eval.dataset import Evidence
from arxiv_agent.eval.relevance import is_relevant, normalize, recall_at_k, reciprocal_rank
from arxiv_agent.retrieval.hits import Hit
from tests.fakes import chunk

QUOTE = "ReAct prompts LLMs to generate both verbal reasoning traces and actions"
EV = Evidence("2210.03629v3", QUOTE)


def hits(*texts: str, paper: str = "2210.03629v3") -> list[Hit]:
    return [Hit(chunk(paper, i, t), 1.0) for i, t in enumerate(texts)]


def test_normalize_folds_case_quotes_dashes_ligatures_and_space() -> None:
    raw = "“Task-speciﬁc”  actions — and\nmore"
    assert normalize(raw) == '"task-specific" actions - and more'


def test_a_chunk_containing_the_quote_is_relevant() -> None:
    c = chunk("2210.03629v3", 0, f"Intro. {QUOTE} in an interleaved manner.")
    assert is_relevant(c, EV)


def test_layout_differences_do_not_matter() -> None:
    c = chunk("2210.03629v3", 0, QUOTE.upper().replace(" ", "\n  "))
    assert is_relevant(c, EV)


def test_a_quote_split_across_chunks_counts_where_most_of_it_is() -> None:
    cut = int(len(QUOTE) * 0.7)
    assert is_relevant(chunk("2210.03629v3", 0, "prefix " + QUOTE[:cut]), EV)
    assert not is_relevant(chunk("2210.03629v3", 1, QUOTE[cut:] + " suffix"), EV)


def test_the_same_text_in_another_paper_is_not_relevant() -> None:
    assert not is_relevant(chunk("2303.11366v4", 0, QUOTE), EV)


def test_reciprocal_rank_is_one_over_the_first_relevant_rank() -> None:
    assert reciprocal_rank(hits("a", "b", QUOTE, QUOTE), [EV], k=5) == pytest.approx(1 / 3)
    assert reciprocal_rank(hits(QUOTE), [EV], k=5) == 1.0


def test_reciprocal_rank_ignores_hits_past_k() -> None:
    assert reciprocal_rank(hits("a", "b", QUOTE), [EV], k=2) == 0.0


def test_any_evidence_counts_for_reciprocal_rank() -> None:
    other = Evidence("2303.11366v4", "Reflexion agents verbally reflect on task feedback signals")
    retrieved = [Hit(chunk("2303.11366v4", 0, other.quote), 1.0)]
    assert reciprocal_rank(retrieved, [EV, other], k=5) == 1.0


def test_recall_counts_evidence_found() -> None:
    other = Evidence("2303.11366v4", "Reflexion agents verbally reflect on task feedback signals")
    assert recall_at_k(hits(QUOTE), [EV, other], k=5) == 0.5
