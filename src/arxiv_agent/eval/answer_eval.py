"""Run the agent over a question set and keep what each run produced."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arxiv_agent.agent.events import Final, ToolStarted
from arxiv_agent.agent.loop import ResearchAgent
from arxiv_agent.eval.dataset import EvalItem


@dataclass(frozen=True)
class AnswerRecord:
    id: str
    question: str
    answer: str
    contexts: tuple[str, ...]  # every passage the run retrieved, in order
    citations: tuple[str, ...]  # verified chunk ids
    unverified_citations: tuple[str, ...]
    tools: tuple[str, ...]  # tool names in call order
    fallback: bool
    reason: str | None
    iterations: int
    latency_s: float
    prompt_tokens: tuple[int, ...] = ()  # input tokens of each model call

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> AnswerRecord:
        row = json.loads(line)
        for key in ("contexts", "citations", "unverified_citations", "tools", "prompt_tokens"):
            row[key] = tuple(row.get(key, ()))
        return cls(**row)


class EvaluationInterruptedError(RuntimeError):
    """The model failed during an evaluation run; the answer was not recorded."""


def load_records(path: Path) -> list[AnswerRecord]:
    if not path.exists():
        return []
    return [AnswerRecord.from_json(line) for line in path.read_text().splitlines() if line]


async def generate_answers(
    agent: ResearchAgent, items: Sequence[EvalItem], out: Path
) -> list[AnswerRecord]:
    """Answer every item, appending each record to ``out`` as it finishes.

    Items already in ``out`` are kept and not asked again, so an interrupted
    run resumes where it stopped. A model failure stops the run without
    recording that item.
    """
    records = {r.id: r for r in load_records(out)}
    for item in items:
        if item.id in records:
            continue
        started = time.perf_counter()
        tools: list[str] = []
        final: Final | None = None
        async for event in agent.run(item.question):
            if isinstance(event, ToolStarted):
                tools.append(event.name)
            elif isinstance(event, Final):
                final = event
        if final is None:  # pragma: no cover - ResearchAgent always ends with a Final
            raise RuntimeError(f"{item.id}: the agent ended without a final event")
        if final.reason == "model_error":
            # An outage or a crash is not a measurement: stop, and let --resume ask again.
            raise EvaluationInterruptedError(
                f"{item.id}: the model failed (see the log); rerun with --resume"
            )
        record = AnswerRecord(
            id=item.id,
            question=item.question,
            answer=final.answer,
            contexts=final.contexts,
            citations=tuple(c.id for c in final.citations),
            unverified_citations=final.unverified_citations,
            tools=tuple(tools),
            fallback=final.fallback,
            reason=final.reason,
            iterations=final.iterations,
            latency_s=round(time.perf_counter() - started, 3),
            prompt_tokens=final.prompt_tokens,
        )
        with out.open("a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")
        records[item.id] = record
    return [records[i.id] for i in items if i.id in records]


def operational_stats(records: Sequence[AnswerRecord]) -> dict[str, Any]:
    """How the runs went, as opposed to how good the answers were."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    latencies = sorted(r.latency_s for r in records)
    p95 = statistics.quantiles(latencies, n=20, method="inclusive")[18] if n > 1 else latencies[0]
    return {
        "n": n,
        "fallback_rate": round(sum(r.fallback for r in records) / n, 4),
        "fallback_reasons": dict(Counter(r.reason for r in records if r.reason)),
        "mean_iterations": round(sum(r.iterations for r in records) / n, 3),
        "latency_p50_s": round(statistics.median(latencies), 2),
        "latency_p95_s": round(p95, 2),
        "cited_rate": round(sum(1 for r in records if r.citations) / n, 4),
        "unverified_citations": sum(len(r.unverified_citations) for r in records),
        "max_prompt_tokens": max((t for r in records for t in r.prompt_tokens), default=0),
    }
