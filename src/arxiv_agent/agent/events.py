"""What a run reports, in order: tool calls, answer tokens, then exactly one Final."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from arxiv_agent.agent.citations import Citation


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


class _Event:
    kind: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict: tuples become lists, citations become objects."""
        fields = asdict(self)  # type: ignore[call-overload]
        return {"kind": self.kind, **_jsonable(fields)}


@dataclass(frozen=True)
class ToolStarted(_Event):
    kind: ClassVar[str] = "tool_started"
    iteration: int
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolFinished(_Event):
    kind: ClassVar[str] = "tool_finished"
    iteration: int
    name: str
    ok: bool
    summary: str


@dataclass(frozen=True)
class Token(_Event):
    """A piece of model text. Text from an iteration that then calls tools is reasoning."""

    kind: ClassVar[str] = "token"
    iteration: int
    text: str


@dataclass(frozen=True)
class Final(_Event):
    kind: ClassVar[str] = "final"
    answer: str
    citations: tuple[Citation, ...]
    unverified_citations: tuple[str, ...]
    fallback: bool
    reason: str | None  # why the run fell back: "budget_exhausted" or "model_error"
    iterations: int
    contexts: tuple[str, ...]  # every passage the run retrieved, in order


AgentEvent = ToolStarted | ToolFinished | Token | Final
