"""The evaluation commands, with their dependencies passed in."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from arxiv_agent.eval.dataset import EvalItem, check_dataset
from arxiv_agent.eval.relevance import is_relevant
from arxiv_agent.eval.retrieval_eval import evaluate_retrieval
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
