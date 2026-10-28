"""The evaluation commands, with their dependencies passed in."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from arxiv_agent.agent.loop import ResearchAgent
from arxiv_agent.eval.answer_eval import (
    AnswerRecord,
    generate_answers,
    load_records,
    operational_stats,
)
from arxiv_agent.eval.dataset import EvalItem, check_dataset
from arxiv_agent.eval.ragas_eval import ScoreRecord
from arxiv_agent.eval.relevance import is_relevant
from arxiv_agent.eval.retrieval_eval import evaluate_retrieval
from arxiv_agent.eval.runs import build_summary, utc_now
from arxiv_agent.retrieval.retriever import Retriever
from arxiv_agent.retrieval.store import PaperStore


def cmd_check_dataset(
    items: Sequence[EvalItem], texts: Mapping[str, str], store: PaperStore, out: TextIO
) -> int:
    problems = check_dataset(items, texts)
    for item in items:
        for ev in item.evidence:
            if not any(is_relevant(c, ev) for c in store.chunks_for(ev.arxiv_id)):
                problems.append(f"{item.id}: no stored chunk holds the quote from {ev.arxiv_id}")
    for problem in problems:
        print(problem, file=out)
    print(f"{len(items)} items, {len(problems)} problems", file=out)
    return 1 if problems else 0


def cmd_eval_retrieval(
    retrievers: Mapping[str, Retriever],
    items: Sequence[EvalItem],
    run_dir: Path,
    meta: Mapping[str, Any],
    out: TextIO,
    k: int = 5,
) -> dict[str, Any]:
    modes = {name: asdict(evaluate_retrieval(r, items, k)) for name, r in retrievers.items()}
    result = {"meta": dict(meta), "modes": modes}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "retrieval.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{'mode':<10} {'MRR@' + str(k):>7} {'Hit@' + str(k):>7} {'Recall@' + str(k):>9}", file=out
    )
    for name, report in modes.items():
        print(
            f"{name:<10} {report['mrr']:>7.3f} {report['hit_rate']:>7.3f} {report['recall']:>9.3f}",
            file=out,
        )
    return result


def cmd_eval_answers(
    agent: ResearchAgent,
    items: Sequence[EvalItem],
    run_dir: Path,
    meta: Mapping[str, Any],
    out: TextIO,
) -> list[AnswerRecord]:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():  # a resumed run keeps the metadata it started with
        meta_path.write_text(json.dumps(dict(meta), indent=2) + "\n")
    records = asyncio.run(generate_answers(agent, items, run_dir / "answers.jsonl"))
    stats = operational_stats(records)
    print(json.dumps(stats, indent=2), file=out)
    return records


Judge = Callable[[Sequence[AnswerRecord], Mapping[str, EvalItem]], list[ScoreRecord]]


def cmd_eval_judge(
    run_dir: Path,
    items: Sequence[EvalItem],
    judge: Judge,
    judge_meta: Mapping[str, Any],
    out: TextIO,
) -> dict[str, Any]:
    records = load_records(run_dir / "answers.jsonl")
    by_id = {i.id: i for i in items}
    scores = judge(records, {r.id: by_id[r.id] for r in records})
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps({"id": s.id, "scores": s.scores}) + "\n" for s in scores)
    )
    meta = {**json.loads((run_dir / "meta.json").read_text()), **judge_meta}
    meta["judged_at"] = utc_now()
    summary = build_summary(meta, records, scores)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for metric, value in summary["ragas"]["mean"].items():
        unscored = summary["ragas"]["unscored"][metric]
        shown = "-" if value is None else f"{value:.3f}"
        print(f"{metric:<18} {shown}  (unscored: {unscored})", file=out)
    return summary
