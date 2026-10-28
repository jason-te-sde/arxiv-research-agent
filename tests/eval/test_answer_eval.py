import json
from pathlib import Path

import pytest

from arxiv_agent.agent.loop import ResearchAgent
from arxiv_agent.agent.tools import ToolContext, build_tools
from arxiv_agent.eval.answer_eval import AnswerRecord, generate_answers, load_records
from arxiv_agent.eval.dataset import EvalItem, Evidence
from arxiv_agent.retrieval.retriever import DenseRetriever
from arxiv_agent.retrieval.store import PaperStore
from tests.agent.scripted import ScriptedChatModel, call, say
from tests.fakes import HashingEmbeddings, chunk

REACT = "2210.03629v3"


def items(n: int) -> list[EvalItem]:
    ev = (Evidence(REACT, "react interleaves reasoning traces and task actions"),)
    return [
        EvalItem(f"t{i:02d}", f"question {i} about react", "ref", ev, "single", True)
        for i in range(n)
    ]


def grounded(text: str) -> list[object]:
    return [call("search_papers", query="react"), say(text)]


def make_agent(tmp_path: Path, script: list[object]) -> ResearchAgent:
    store = PaperStore(tmp_path / "c", HashingEmbeddings(), "ans")
    store.replace_paper([chunk(REACT, 0, "react interleaves reasoning traces", title="ReAct")])
    model = ScriptedChatModel(script=script)
    retriever = DenseRetriever(store)
    return ResearchAgent(
        model, build_tools, lambda: ToolContext(retriever, store, model), retriever
    )


async def test_one_record_per_item_with_what_the_run_used(tmp_path: Path) -> None:
    agent = make_agent(
        tmp_path, [call("search_papers", query="react"), say(f"Interleaves [{REACT}#0].")]
    )
    out = tmp_path / "answers.jsonl"
    [record] = await generate_answers(agent, items(1), out)
    assert record.id == "t00"
    assert record.answer == f"Interleaves [{REACT}#0]."
    assert record.contexts == ("ReAct\nreact interleaves reasoning traces",)
    assert record.citations == (f"{REACT}#0",)
    assert record.tools == ("search_papers",)
    assert record.iterations == 2
    assert not record.fallback
    assert record.latency_s >= 0


async def test_records_are_appended_as_they_finish(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, [*grounded("one"), *grounded("two")])
    out = tmp_path / "answers.jsonl"
    await generate_answers(agent, items(2), out)
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["t00", "t01"]
    assert [r.id for r in load_records(out)] == ["t00", "t01"]


async def test_a_rerun_resumes_after_the_answered_items(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    await generate_answers(make_agent(tmp_path / "a", grounded("one")), items(1), out)
    agent = make_agent(tmp_path / "b", grounded("two"))
    records = await generate_answers(agent, items(2), out)
    assert [r.answer for r in records] == ["one", "two"]


def test_record_round_trips_through_json() -> None:
    record = AnswerRecord(
        id="t01",
        question="q",
        answer="a",
        contexts=("c",),
        citations=("x#1",),
        unverified_citations=(),
        tools=("search_papers",),
        fallback=True,
        reason="budget_exhausted",
        iterations=3,
        latency_s=1.5,
    )
    assert AnswerRecord.from_json(record.to_json()) == record


@pytest.mark.parametrize("n", [0])
async def test_no_items_no_records(tmp_path: Path, n: int) -> None:
    assert await generate_answers(make_agent(tmp_path, []), items(n), tmp_path / "a.jsonl") == []


def test_operational_stats() -> None:
    from arxiv_agent.eval.answer_eval import operational_stats

    def rec(i: int, fallback: bool, latency: float, cites: tuple[str, ...]) -> AnswerRecord:
        reason = "budget_exhausted" if fallback else None
        return AnswerRecord(
            f"t{i}",
            "q",
            "a",
            (),
            cites,
            ("bad#1",) if i == 0 else (),
            (),
            fallback,
            reason,
            3 if fallback else 2,
            latency,
        )

    stats = operational_stats(
        [
            rec(0, False, 10.0, ("x#1",)),
            rec(1, True, 30.0, ()),
            rec(2, False, 20.0, ("y#2",)),
            rec(3, False, 40.0, ("z#3",)),
        ]
    )
    assert stats["n"] == 4
    assert stats["fallback_rate"] == 0.25
    assert stats["fallback_reasons"] == {"budget_exhausted": 1}
    assert stats["mean_iterations"] == 2.25
    assert stats["latency_p50_s"] == 25.0
    assert stats["cited_rate"] == 0.75
    assert stats["unverified_citations"] == 1


def test_records_written_before_prompt_sizes_existed_still_load() -> None:
    old = (
        '{"id": "t01", "question": "q", "answer": "a", "contexts": [], "citations": [], '
        '"unverified_citations": [], "tools": [], "fallback": false, "reason": null, '
        '"iterations": 1, "latency_s": 1.0}'
    )
    assert AnswerRecord.from_json(old).prompt_tokens == ()


def test_stats_report_the_largest_prompt() -> None:
    from arxiv_agent.eval.answer_eval import operational_stats

    base = AnswerRecord("t", "q", "a", (), (), (), (), False, None, 2, 1.0)
    records = [
        AnswerRecord(**{**base.__dict__, "prompt_tokens": (1800, 4200)}),
        AnswerRecord(**{**base.__dict__, "prompt_tokens": (2100, 6100)}),
    ]
    assert operational_stats(records)["max_prompt_tokens"] == 6100
