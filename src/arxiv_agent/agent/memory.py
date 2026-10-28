"""Conversation memory: a sliding window of turns plus a running summary of older ones."""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


Summarize = Callable[[str, Sequence[Turn]], Awaitable[str]]


class ConversationMemory:
    def __init__(self, window_turns: int, summarize: Summarize | None) -> None:
        if window_turns < 1:
            raise ValueError("window_turns must be at least 1")
        self._window = window_turns
        self._summarize = summarize
        self._turns: deque[Turn] = deque()
        self.summary = ""

    def messages(self) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for turn in self._turns:
            out += [HumanMessage(turn.question), AIMessage(turn.answer)]
        return out

    async def add(self, question: str, answer: str) -> None:
        self._turns.append(Turn(question, answer))
        evicted = [self._turns.popleft() for _ in range(len(self._turns) - self._window)]
        if not evicted or self._summarize is None:
            return
        try:
            self.summary = await self._summarize(self.summary, evicted)
        except Exception:
            # The evicted turns are lost from the summary, but the conversation goes on.
            log.warning("conversation summary failed; keeping the previous one", exc_info=True)


class SessionStore:
    """In-process sessions, least recently used evicted past ``max_sessions``."""

    def __init__(self, factory: Callable[[], ConversationMemory], max_sessions: int = 1000) -> None:
        self._factory = factory
        self._max = max_sessions
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()

    def get(self, session_id: str) -> ConversationMemory:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
        else:
            self._sessions[session_id] = self._factory()
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        return self._sessions[session_id]

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._sessions
