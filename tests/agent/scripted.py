"""A chat model that replays a script, so agent behaviour can be tested exactly."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


def call(name: str, call_id: str = "c1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def broken_call(name: str, raw_args: str, call_id: str = "c1") -> AIMessage:
    """A tool call whose arguments are not valid JSON, as a small model can emit."""
    return AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": name,
                "args": raw_args,
                "id": call_id,
                "error": None,
                "type": "invalid_tool_call",
            }
        ],
    )


class ScriptedChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Any, not AIMessage | Exception: pydantic would try to coerce exceptions into messages.
    script: list[Any]
    inputs: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.inputs.append(list(messages))
        if self._cursor >= len(self.script):
            raise AssertionError(f"script exhausted after {self._cursor} calls")
        step = self.script[self._cursor]
        self._cursor += 1
        if isinstance(step, Exception):
            raise step
        assert isinstance(step, AIMessage)
        return step

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        text = message.content if isinstance(message.content, str) else ""
        if not (
            text
            or message.tool_calls
            or message.invalid_tool_calls
            or message.response_metadata
            or message.usage_metadata
        ):
            # Real providers always send at least one chunk, even for an empty reply.
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
        for start in range(0, len(text), 4):
            yield ChatGenerationChunk(message=AIMessageChunk(content=text[start : start + 4]))
        if message.invalid_tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        tool_call_chunk(name=tc["name"], args=tc["args"], id=tc["id"], index=i)
                        for i, tc in enumerate(message.invalid_tool_calls)
                    ],
                )
            )
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        tool_call_chunk(
                            name=tc["name"], args=json.dumps(tc["args"]), id=tc["id"], index=i
                        )
                        for i, tc in enumerate(message.tool_calls)
                    ],
                )
            )
        if message.response_metadata or message.usage_metadata:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    response_metadata=message.response_metadata,
                    usage_metadata=message.usage_metadata,
                )
            )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    @property
    def calls(self) -> int:
        return self._cursor
