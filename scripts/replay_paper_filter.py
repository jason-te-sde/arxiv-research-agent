"""Replay an answer run's search_papers calls under each way of handling arxiv_id.

Each call's query and arxiv_id come from the run's answers.jsonl and are
searched again on the current index; every top 5 is judged against the
question's evidence quotes (ADR 4). No model is involved, so the comparison
holds the agent's queries fixed. ADR 6 reports its output for
eval/runs/20261029T124034Z-answers-dev.

    python scripts/replay_paper_filter.py eval/runs/<run>-answers-dev
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from arxiv_agent.agent.tools import ToolContext, _merge
from arxiv_agent.config import Settings
from arxiv_agent.eval.dataset import load_items
from arxiv_agent.eval.relevance import first_relevant_rank
from arxiv_agent.retrieval.hits import Hit
from arxiv_agent.retrieval.retriever import Retriever
from arxiv_agent.runtime import make_retriever, open_store

Strategy = Callable[[Retriever, str, str | None], list[Hit]]


def restrict(retriever: Retriever, query: str, paper: str | None) -> list[Hit]:
    """Search only the named paper (before ADR 6)."""
    return retriever.search(query, paper)


def ignore(retriever: Retriever, query: str, _paper: str | None) -> list[Hit]:
    """Search the whole library whatever the call names."""
    return retriever.search(query)


def favor(retriever: Retriever, query: str, paper: str | None) -> list[Hit]:
    """What search_papers does (ADR 6)."""
    everywhere = retriever.search(query)
    return _merge(retriever.search(query, paper), everywhere) if paper else everywhere


def reserve(retriever: Retriever, query: str, paper: str | None) -> list[Hit]:
    """As favor, but the named paper's best passage always takes the last place."""
    hits = favor(retriever, query, paper)
    if paper is None or any(h.chunk.arxiv_id == paper for h in hits):
        return hits
    inside = retriever.search(query, paper)
    return [*hits[:-1], inside[0]] if inside else hits


STRATEGIES: dict[str, Strategy] = {
    "restrict": restrict,
    "ignore": ignore,
    "favor": favor,
    "reserve": reserve,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="an answer run directory")
    run: Path = parser.parse_args().run

    settings = Settings.from_env(os.environ)
    store = open_store(settings)
    retriever = make_retriever(settings, store, "two-stage")
    # resolve() reads only the store; no tool here calls the model.
    ctx = ToolContext(retriever, store, model=None)  # type: ignore[arg-type]
    meta = json.loads((run / "meta.json").read_text())
    items = {i.id: i for i in load_items(Path("eval/questions") / meta["dataset"])}
    records = [json.loads(line) for line in (run / "answers.jsonl").read_text().splitlines()]

    for name, strategy in STRATEGIES.items():
        calls = hits_found = reached = 0
        reciprocal = 0.0
        for record in records:
            item = items[record["id"]]
            found = False
            for tool, args in zip(record["tools"], record["tool_args"], strict=True):
                if tool != "search_papers":
                    continue
                raw = args.get("arxiv_id") or None
                try:
                    paper = ctx.resolve(raw) if raw else None
                except ValueError:
                    paper = None
                if raw and paper is None and strategy is restrict:
                    continue  # the tool answered with an error and searched nothing
                hits = strategy(retriever, args.get("query", ""), paper)
                rank = first_relevant_rank(hits, item.evidence, settings.k_final)
                calls += 1
                hits_found += rank is not None
                reciprocal += 1 / rank if rank else 0.0
                found |= rank is not None
            reached += found
        per_call = max(calls, 1)
        print(
            f"{name:8} calls={calls} hit@5/call={hits_found / per_call:.3f} "
            f"MRR@5/call={reciprocal / per_call:.3f} "
            f"questions with an evidence passage in a top 5={reached}/{len(records)}"
        )


if __name__ == "__main__":
    main()
