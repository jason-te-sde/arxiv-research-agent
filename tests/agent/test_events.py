from arxiv_agent.agent.citations import Citation
from arxiv_agent.agent.events import Final, Token, ToolFinished, ToolStarted


def test_events_serialize_with_their_kind() -> None:
    assert ToolStarted(1, "search_papers", {"query": "q"}).to_dict() == {
        "kind": "tool_started",
        "iteration": 1,
        "name": "search_papers",
        "args": {"query": "q"},
    }
    assert ToolFinished(1, "search_papers", True, "5 passages").to_dict()["kind"] == "tool_finished"
    assert Token(2, "Re").to_dict() == {"kind": "token", "iteration": 2, "text": "Re"}


def test_final_serializes_citations_as_objects() -> None:
    final = Final(
        answer="x [a#1]",
        citations=(Citation("a#1", "a", "T", 2),),
        unverified_citations=(),
        fallback=False,
        reason=None,
        iterations=2,
        contexts=("ctx",),
    )
    d = final.to_dict()
    assert d["kind"] == "final"
    assert d["citations"] == [{"id": "a#1", "arxiv_id": "a", "title": "T", "page": 2}]
    assert d["contexts"] == ["ctx"]
