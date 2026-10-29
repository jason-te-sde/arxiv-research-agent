"""The question sets: JSONL items whose evidence is quoted verbatim from the papers.

Each item names the passages that answer it by quoting them, which is what
lets retrieval be scored without any model: a retrieved chunk is relevant when
it contains the quote (see relevance.py).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from arxiv_agent.ingest.arxiv_client import base_id, normalize_arxiv_id

Kind = Literal["single", "multi", "numeric"]

# A quote shorter than this matches too much text to identify a passage.
MIN_QUOTE_CHARS = 40
# Longer quotes could straddle two chunks without either holding most of it.
MAX_QUOTE_CHARS = 400


@dataclass(frozen=True)
class Evidence:
    arxiv_id: str
    quote: str


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    reference_answer: str
    evidence: tuple[Evidence, ...]
    kind: Kind
    names_method: bool  # whether the question names the paper's method or title


def load_items(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = _parse(json.loads(line))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    return items


def _parse(row: dict[str, object]) -> EvalItem:
    kind = str(row["kind"])
    if kind not in get_args(Kind):
        raise ValueError(f"kind must be one of {get_args(Kind)}, got {kind!r}")
    raw_evidence = row["evidence"]
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError("evidence must be a non-empty list")
    evidence = []
    for ev in raw_evidence:
        arxiv_id = normalize_arxiv_id(str(ev["arxiv_id"]))
        if arxiv_id == base_id(arxiv_id):
            raise ValueError(f"evidence id {arxiv_id} must pin a version")
        quote = " ".join(str(ev["quote"]).split())
        if not MIN_QUOTE_CHARS <= len(quote) <= MAX_QUOTE_CHARS:
            raise ValueError(
                f"quote must be {MIN_QUOTE_CHARS}-{MAX_QUOTE_CHARS} characters: {quote!r}"
            )
        evidence.append(Evidence(arxiv_id, quote))
    question = str(row["question"]).strip()
    if not question:
        raise ValueError("question is empty")
    return EvalItem(
        id=str(row["id"]),
        question=question,
        reference_answer=str(row["reference_answer"]).strip(),
        evidence=tuple(evidence),
        kind=cast(Kind, kind),
        names_method=bool(row["names_method"]),
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_dataset(items: Sequence[EvalItem], texts: Mapping[str, str]) -> list[str]:
    """Problems with the evidence: each quote must occur in its paper's extracted text."""
    from arxiv_agent.eval.relevance import normalize

    normalized = {k: normalize(v) for k, v in texts.items()}
    problems = []
    for item in items:
        for ev in item.evidence:
            text = normalized.get(ev.arxiv_id)
            if text is None:
                problems.append(f"{item.id}: no text for {ev.arxiv_id} (is it ingested?)")
            elif normalize(ev.quote) not in text:
                problems.append(f"{item.id}: quote not found in {ev.arxiv_id}: {ev.quote!r}")
    return problems
