from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from arxiv_agent.agent.memory import ConversationMemory, SessionStore, Turn


class Summarizer:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, list[Turn]]] = []
        self.fail = fail

    async def __call__(self, previous: str, evicted: Sequence[Turn]) -> str:
        self.calls.append((previous, list(evicted)))
        if self.fail:
            raise RuntimeError("model down")
        return previous + "|" + ",".join(t.question for t in evicted)


async def fill(memory: ConversationMemory, n: int) -> None:
    for i in range(n):
        await memory.add(f"q{i}", f"a{i}")


async def test_messages_alternate_question_and_answer() -> None:
    memory = ConversationMemory(window_turns=10, summarize=Summarizer())
    await fill(memory, 2)
    assert memory.messages() == [
        HumanMessage("q0"),
        AIMessage("a0"),
        HumanMessage("q1"),
        AIMessage("a1"),
    ]


async def test_window_keeps_the_last_ten_turns_and_summarizes_the_rest() -> None:
    summarizer = Summarizer()
    memory = ConversationMemory(window_turns=10, summarize=summarizer)
    await fill(memory, 11)
    assert len(memory.messages()) == 20
    assert memory.messages()[0] == HumanMessage("q1")
    assert summarizer.calls == [("", [Turn("q0", "a0")])]
    assert memory.summary == "|q0"


async def test_summary_accumulates() -> None:
    memory = ConversationMemory(window_turns=1, summarize=Summarizer())
    await fill(memory, 3)
    assert memory.summary == "|q0|q1"


async def test_failed_summary_keeps_the_old_one_and_still_evicts() -> None:
    memory = ConversationMemory(window_turns=1, summarize=Summarizer())
    await fill(memory, 2)
    memory._summarize = Summarizer(fail=True)  # the model goes down mid-session
    await memory.add("q2", "a2")
    assert memory.summary == "|q0"
    assert memory.messages() == [HumanMessage("q2"), AIMessage("a2")]


async def test_without_a_summarizer_old_turns_are_dropped() -> None:
    memory = ConversationMemory(window_turns=1, summarize=None)
    await fill(memory, 3)
    assert memory.summary == ""
    assert len(memory.messages()) == 2


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window"):
        ConversationMemory(window_turns=0, summarize=None)


def test_session_store_reuses_a_session() -> None:
    store = SessionStore(lambda: ConversationMemory(10, None), max_sessions=2)
    assert store.get("a") is store.get("a")


def test_session_store_evicts_the_least_recently_used() -> None:
    store = SessionStore(lambda: ConversationMemory(10, None), max_sessions=2)
    a = store.get("a")
    store.get("b")
    store.get("a")
    store.get("c")
    assert store.get("a") is a
    assert len(store) == 2
    assert "b" not in store
