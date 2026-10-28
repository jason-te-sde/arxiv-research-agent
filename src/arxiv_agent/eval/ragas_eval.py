"""Answer quality with RAGAS: context precision and recall, faithfulness, relevancy.

The judge is a chat model (by default a local llama3.1:8b, a different family
from the qwen3 agent). Citation markers are removed from answers first: they
are pointers, not claims, and a judge would otherwise try to verify them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from arxiv_agent.eval.answer_eval import AnswerRecord
from arxiv_agent.eval.dataset import EvalItem

METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")

# RAGAS result column for each metric.
_COLUMNS = {
    "context_precision": "llm_context_precision_with_reference",
    "context_recall": "context_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
}

_CITATION = re.compile(r"\s*\[[^\[\]]*#\d+[^\[\]]*\]")


@dataclass(frozen=True)
class ScoreRecord:
    id: str
    scores: dict[str, float | None]  # None or NaN: the judge produced no score


def to_sample(record: AnswerRecord, item: EvalItem) -> dict[str, Any]:
    return {
        "user_input": item.question,
        "response": _CITATION.sub("", record.answer).strip(),
        "retrieved_contexts": list(record.contexts),
        "reference": item.reference_answer,
    }


def _scored(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def summarize(scores: Sequence[ScoreRecord]) -> dict[str, Any]:
    """Means over scored values; unscored values are counted, never averaged in as zero."""
    mean: dict[str, float | None] = {}
    unscored: dict[str, int] = {}
    for metric in METRICS:
        values = [s.scores.get(metric) for s in scores]
        good = [v for v in values if _scored(v)]
        mean[metric] = round(sum(good) / len(good), 4) if good else None  # type: ignore[arg-type]
        unscored[metric] = len(values) - len(good)
    return {"n": len(scores), "mean": mean, "unscored": unscored}


def judge(
    records: Sequence[AnswerRecord],
    items: Mapping[str, EvalItem],
    llm: Any,
    embeddings: Any,
    timeout_s: int = 900,
) -> list[ScoreRecord]:  # pragma: no cover - needs a judge model; see docs/evaluation.md
    """Score every record with RAGAS. ``llm`` is a LangChain chat model."""
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    # An answer given without retrieving anything has no context to be precise,
    # complete or faithful to: it scores zero there and is judged only on relevancy.
    with_context = [r for r in records if r.contexts]
    without = [r for r in records if not r.contexts]
    run_config = RunConfig(timeout=timeout_s, max_workers=1, max_retries=3)
    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    def run(rs: Sequence[AnswerRecord], metrics: list[Any]) -> list[dict[str, Any]]:
        if not rs:
            return []
        dataset = EvaluationDataset.from_list([to_sample(r, items[r.id]) for r in rs])
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=False,
        )
        if not isinstance(result, EvaluationResult):  # only returned when return_executor=True
            raise TypeError(f"unexpected RAGAS result {type(result).__name__}")
        return list(result.scores)

    full = run(
        with_context,
        [
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
        ],
    )
    relevancy_only = run(without, [ResponseRelevancy()])
    out = []
    for record, row in zip(with_context, full, strict=True):
        out.append(ScoreRecord(record.id, {m: _float(row.get(_COLUMNS[m])) for m in METRICS}))
    for record, row in zip(without, relevancy_only, strict=True):
        scores: dict[str, float | None] = dict.fromkeys(METRICS, 0.0)
        scores["answer_relevancy"] = _float(row.get(_COLUMNS["answer_relevancy"]))
        out.append(ScoreRecord(record.id, scores))
    order = {r.id: n for n, r in enumerate(records)}
    return sorted(out, key=lambda s: order[s.id])


def _float(value: Any) -> float | None:
    return None if value is None else float(value)
