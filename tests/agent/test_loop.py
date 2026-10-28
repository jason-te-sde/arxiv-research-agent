from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from arxiv_agent.agent.events import AgentEvent, Final, Token, ToolFinished, ToolStarted
from arxiv_agent.agent.loop import ResearchAgent
from arxiv_agent.agent.memory import ConversationMemory, Turn
from arxiv_agent.agent.tools import ToolContext, build_tools
from arxiv_agent.retrieval.retriever import DenseRetriever
from arxiv_agent.retrieval.store import PaperStore
from tests.agent.scripted import ScriptedChatModel, call, say
from tests.fakes import HashingEmbeddings, chunk

REACT = "2210.03629v3"


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    s = PaperStore(tmp_path / "chroma", HashingEmbeddings(), "loop")
    s.replace_paper(
        [
            chunk(REACT, 0, "react interleaves reasoning traces and actions", title="ReAct"),
            chunk(REACT, 1, "react improves hotpotqa and fever results", title="ReAct"),
        ]
    )
    return s


def agent(store: PaperStore, model: ScriptedChatModel, max_iterations: int = 3) -> ResearchAgent:
    retriever = DenseRetriever(store, k=2)
    return ResearchAgent(
        model=model,
        tool_factory=build_tools,
        ctx_factory=lambda: ToolContext(retriever=retriever, store=store, model=model),
        retriever=retriever,
        max_iterations=max_iterations,
    )


async def run(
    a: ResearchAgent, question: str, memory: ConversationMemory | None = None
) -> list[AgentEvent]:
    return [e async for e in a.run(question, memory)]


def final(events: Sequence[AgentEvent]) -> Final:
    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert events[-1] is finals[0]
    return finals[0]


async def test_search_then_answer(store: PaperStore) -> None:
    model = ScriptedChatModel(
        script=[
            call("search_papers", query="react reasoning actions"),
            say(f"ReAct interleaves reasoning and acting [{REACT}#0]."),
        ]
    )
    events = await run(agent(store, model), "What is ReAct?")
    result = final(events)
    assert not result.fallback
    assert result.iterations == 2
    assert [c.id for c in result.citations] == [f"{REACT}#0"]
    assert result.unverified_citations == ()
    assert result.contexts[0].startswith("ReAct\n")
    started = [e for e in events if isinstance(e, ToolStarted)]
    assert [(e.name, e.args) for e in started] == [
        ("search_papers", {"query": "react reasoning actions"})
    ]
    assert model.bound_tools == [
        "search_papers",
        "summarize_paper",
        "fetch_paper",
        "compare_papers",
    ]


async def test_tokens_of_the_last_iteration_are_the_answer(store: PaperStore) -> None:
    model = ScriptedChatModel(
        script=[call("search_papers", query="react"), say(f"An answer [{REACT}#1].")]
    )
    events = await run(agent(store, model), "q")
    tokens = "".join(e.text for e in events if isinstance(e, Token) and e.iteration == 2)
    assert tokens == final(events).answer


async def test_exhausted_budget_falls_back_to_direct_retrieval(store: PaperStore) -> None:
    model = ScriptedChatModel(
        script=[
            call("search_papers", "c1", query="a"),
            call("search_papers", "c2", query="b"),
            call("search_papers", "c3", query="c"),
            say(f"From the passages: ReAct helps [{REACT}#1]."),
        ]
    )
    result = final(await run(agent(store, model), "How does ReAct do on HotpotQA?"))
    assert result.fallback
    assert result.reason == "budget_exhausted"
    assert result.iterations == 3
    assert model.calls == 4
    assert "using only the passages" in model.inputs[3][-1].text
    assert result.answer.startswith("From the passages")


async def test_model_error_answers_from_passages_without_a_model(store: PaperStore) -> None:
    model = ScriptedChatModel(script=[ConnectionError("ollama is not running")])
    result = final(await run(agent(store, model), "react reasoning"))
    assert result.fallback
    assert result.reason == "model_error"
    assert model.calls == 1
    assert f"[{REACT}#0]" in result.answer
    assert result.citations


async def test_failed_fallback_answer_becomes_extractive(store: PaperStore) -> None:
    model = ScriptedChatModel(
        script=[
            call("search_papers", "c1", query="a"),
            call("search_papers", "c2", query="b"),
            call("search_papers", "c3", query="c"),
            TimeoutError("too slow"),
        ]
    )
    result = final(await run(agent(store, model), "react reasoning"))
    assert result.fallback
    assert result.reason == "budget_exhausted"
    assert f"[{REACT}#" in result.answer


async def test_empty_answer_falls_back(store: PaperStore) -> None:
    model = ScriptedChatModel(script=[say(""), say(f"Fallback answer [{REACT}#0].")])
    result = final(await run(agent(store, model), "react"))
    assert result.reason == "empty_answer"
    assert result.answer.startswith("Fallback answer")


async def test_refusal_is_a_model_error(store: PaperStore) -> None:
    refused = AIMessage(content="", response_metadata={"stop_reason": "refusal"})
    model = ScriptedChatModel(script=[refused])
    result = final(await run(agent(store, model), "react"))
    assert result.reason == "model_error"


async def test_unknown_tool_is_an_observation_not_a_crash(store: PaperStore) -> None:
    model = ScriptedChatModel(script=[call("browse_web", url="x"), say("I could not find that.")])
    events = await run(agent(store, model), "q")
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].ok is False
    assert "Unknown tool" in model.inputs[1][-1].text
    assert not final(events).fallback


async def test_invalid_arguments_are_an_observation(store: PaperStore) -> None:
    model = ScriptedChatModel(script=[call("search_papers", wrong="x"), say("No answer.")])
    events = await run(agent(store, model), "q")
    assert [e.ok for e in events if isinstance(e, ToolFinished)] == [False]
    assert not final(events).fallback


async def test_citation_the_tools_never_returned_is_unverified(store: PaperStore) -> None:
    model = ScriptedChatModel(
        script=[call("search_papers", query="react"), say("Claim [2401.15884v3#9].")]
    )
    result = final(await run(agent(store, model), "q"))
    assert result.citations == ()
    assert result.unverified_citations == ("2401.15884v3#9",)


async def test_system_prompt_lists_the_papers_and_the_budget(store: PaperStore) -> None:
    model = ScriptedChatModel(script=[say("No.")])
    await run(agent(store, model, max_iterations=3), "q")
    system = model.inputs[0][0]
    assert isinstance(system, SystemMessage)
    assert f"{REACT}: ReAct" in system.text
    assert "at most 3 model turns" in system.text


async def test_memory_carries_the_conversation(store: PaperStore) -> None:
    memory = ConversationMemory(window_turns=10, summarize=None)
    memory.summary = "The user studies ReAct."
    model = ScriptedChatModel(script=[say("First."), say("Second.")])
    a = agent(store, model)
    await run(a, "first question", memory)
    await run(a, "second question", memory)
    second = model.inputs[1]
    assert [m.text for m in second[1:]] == ["first question", "First.", "second question"]
    assert "The user studies ReAct." in second[0].text
    assert memory.messages()[-1].text == "Second."


async def test_memory_is_updated_before_the_final_event(store: PaperStore) -> None:
    seen: list[Turn] = []

    async def summarize(previous: str, evicted: Sequence[Turn]) -> str:
        seen.extend(evicted)
        return previous

    memory = ConversationMemory(window_turns=1, summarize=summarize)
    model = ScriptedChatModel(script=[say("One."), say("Two.")])
    a = agent(store, model)
    await run(a, "q1", memory)
    async for event in a.run("q2", memory):
        if isinstance(event, Final):
            break  # a client that stops reading at the final event
    assert seen == [Turn("q1", "One.")]
