from arxiv_agent.agent.citations import Citation, extract_citations, verify_citations
from tests.fakes import chunk


def test_extracts_bracketed_chunk_ids_in_order_without_duplicates() -> None:
    text = (
        "ReAct interleaves reasoning and acting [2210.03629v3#4]. Toolformer learns tools "
        "[2302.04761v1#0; 2210.03629v3#4] and [2302.04761v1#2, 2302.04761v1#0]."
    )
    assert extract_citations(text) == ["2210.03629v3#4", "2302.04761v1#0", "2302.04761v1#2"]


def test_accepts_local_upload_ids() -> None:
    assert extract_citations("see [local-0123456789ab#7]") == ["local-0123456789ab#7"]


def test_ignores_brackets_that_are_not_chunk_ids() -> None:
    assert extract_citations("a list [1], a range [2, 3] and [see above]") == []


def test_verify_splits_seen_from_unseen() -> None:
    seen = {"2210.03629v3#4": chunk("2210.03629v3", 4, "body", title="ReAct", page=3)}
    verified, unverified = verify_citations(["2210.03629v3#4", "9999.99999v1#0"], seen)
    assert verified == [Citation("2210.03629v3#4", "2210.03629v3", "ReAct", 3)]
    assert unverified == ["9999.99999v1#0"]
