"""The ReAct loop: reason, call tools, observe, answer, within a fixed budget.

Each iteration is one model call. A call that returns tool calls is a reasoning
step: the tools run and their observations go back to the model. A call that
returns only text is the answer. If the budget runs out, the model fails, or the
answer is empty, the run falls back to direct retrieval: search with the
question itself and answer from those passages in one call, or, if no model is
available, return the passages themselves. Every run ends in exactly one Final.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from arxiv_agent.agent import prompts
from arxiv_agent.agent.citations import extract_citations, verify_citations
from arxiv_agent.agent.events import AgentEvent, Final, Token, ToolFinished, ToolStarted
from arxiv_agent.agent.memory import ConversationMemory, Summarize, Turn
from arxiv_agent.agent.tools import ToolContext, format_hits
from arxiv_agent.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

_SUMMARY_CHARS = 200


class ModelRefusalError(Exception):
    """The provider declined to answer (Anthropic stop_reason "refusal")."""


class ResearchAgent:
    def __init__(
        self,
        model: BaseChatModel,
        tool_factory: Callable[[ToolContext], list[BaseTool]],
        ctx_factory: Callable[[], ToolContext],
        retriever: Retriever,
        max_iterations: int = 3,
    ) -> None:
        self._model = model
        self._tool_factory = tool_factory
        self._ctx_factory = ctx_factory
        self._retriever = retriever
        self._max_iterations = max_iterations

    async def run(
        self, question: str, memory: ConversationMemory | None = None
    ) -> AsyncIterator[AgentEvent]:
        ctx = self._ctx_factory()
        tools = {t.name: t for t in self._tool_factory(ctx)}
        bound = self._model.bind_tools(list(tools.values()))
        messages: list[BaseMessage] = [
            SystemMessage(await self._system_prompt(ctx, memory)),
            *(memory.messages() if memory else []),
            HumanMessage(question),
        ]

        reason = "budget_exhausted"
        iteration = 0
        try:
            while iteration < self._max_iterations:
                iteration += 1
                reply: AIMessageChunk | None = None
                async for piece in bound.astream(messages):
                    reply = _accumulate(reply, piece)
                    if piece.text:
                        yield Token(iteration, piece.text)
                message = _finish(reply)
                messages.append(message)
                calls = [*message.tool_calls, *_invalid_calls(message)]
                if not calls:
                    if message.text.strip():
                        final = _final(message.text, ctx, iteration, fallback=False, reason=None)
                        await _remember(memory, question, final.answer)
                        yield final
                        return
                    reason = "empty_answer"
                    break
                for call in calls:
                    yield ToolStarted(iteration, call["name"], dict(call.get("args") or {}))
                    observation, ok = await _run_tool(tools, call)
                    yield ToolFinished(iteration, call["name"], ok, observation[:_SUMMARY_CHARS])
                    messages.append(
                        ToolMessage(observation, tool_call_id=call["id"], name=call["name"])
                    )
        except Exception:
            log.warning("agent model call failed; falling back to retrieval", exc_info=True)
            reason = "model_error"

        async for event in self._fallback(question, ctx, memory, reason, iteration):
            yield event

    async def _fallback(
        self,
        question: str,
        ctx: ToolContext,
        memory: ConversationMemory | None,
        reason: str,
        iterations: int,
    ) -> AsyncIterator[AgentEvent]:
        hits = await asyncio.to_thread(self._retriever.search, question)
        ctx.record([h.chunk for h in hits])
        answer = ""
        if not hits:
            answer = "I found no passages about this in the ingested papers."
        elif reason != "model_error":
            prompt = prompts.FALLBACK.format(passages=format_hits(hits), question=question)
            try:
                reply: AIMessageChunk | None = None
                async for piece in self._model.astream([HumanMessage(prompt)]):
                    reply = _accumulate(reply, piece)
                    if piece.text:
                        yield Token(iterations + 1, piece.text)
                answer = _finish(reply).text.strip()
            except Exception:
                log.warning("fallback answer failed; returning passages", exc_info=True)
                answer = ""
        if not answer:
            answer = (
                "The model could not answer, so these are the most relevant passages:\n\n"
                + "\n\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)
            )
        final = _final(answer, ctx, iterations, fallback=True, reason=reason)
        await _remember(memory, question, final.answer)
        yield final

    async def _system_prompt(self, ctx: ToolContext, memory: ConversationMemory | None) -> str:
        papers = await asyncio.to_thread(ctx.store.papers)
        catalog = "\n".join(f"{p.arxiv_id}: {p.title}" for p in papers) or "(none yet)"
        prompt = prompts.SYSTEM.format(max_iterations=self._max_iterations, catalog=catalog)
        if memory and memory.summary:
            prompt += prompts.MEMORY.format(summary=memory.summary)
        return prompt


def summarizer_for(model: BaseChatModel) -> Summarize:
    """Conversation-memory summarization with ``model``."""

    async def summarize(previous: str, evicted: Sequence[Turn]) -> str:
        turns = "\n\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in evicted)
        prompt = prompts.CONVERSATION_SUMMARY.format(summary=previous or "(empty)", turns=turns)
        reply = await model.ainvoke([HumanMessage(prompt)])
        return reply.text.strip() or previous

    return summarize


def _accumulate(reply: AIMessageChunk | None, piece: BaseMessage) -> AIMessageChunk:
    if not isinstance(piece, AIMessageChunk):
        raise TypeError(f"expected a streamed AIMessageChunk, got {type(piece).__name__}")
    return piece if reply is None else reply + piece


def _finish(reply: AIMessageChunk | None) -> AIMessage:
    """The streamed chunks as one message; a provider refusal is a model error."""
    if reply is None:
        return AIMessage(content="")
    if reply.response_metadata.get("stop_reason") == "refusal":
        raise ModelRefusalError(str(reply.response_metadata.get("stop_details")))
    return AIMessage(
        content=reply.content,
        tool_calls=reply.tool_calls,
        invalid_tool_calls=reply.invalid_tool_calls,
        response_metadata=reply.response_metadata,
        id=reply.id,
    )


def _invalid_calls(message: AIMessage) -> list[dict[str, Any]]:
    # Calls whose arguments were not valid JSON still need an answer per call id.
    return [
        {"name": c.get("name") or "unknown", "args": {}, "id": c.get("id") or "", "error": True}
        for c in message.invalid_tool_calls
    ]


async def _run_tool(tools: dict[str, BaseTool], call: Any) -> tuple[str, bool]:
    name = call["name"]
    if call.get("error"):
        return f"The arguments for {name} were not valid JSON. Try again.", False
    tool = tools.get(name)
    if tool is None:
        return f"Unknown tool {name!r}. Available tools: {', '.join(tools)}.", False
    try:
        return str(await tool.ainvoke(call["args"])), True
    except Exception as exc:
        return f"{name} failed: {exc}", False


def _final(
    answer: str, ctx: ToolContext, iterations: int, *, fallback: bool, reason: str | None
) -> Final:
    citations, unverified = verify_citations(extract_citations(answer), ctx.seen)
    return Final(
        answer=answer,
        citations=tuple(citations),
        unverified_citations=tuple(unverified),
        fallback=fallback,
        reason=reason,
        iterations=iterations,
        contexts=tuple(ctx.contexts),
    )


async def _remember(memory: ConversationMemory | None, question: str, answer: str) -> None:
    if memory is not None:
        await memory.add(question, answer)
