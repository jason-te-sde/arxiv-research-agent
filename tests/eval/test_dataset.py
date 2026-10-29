import json
from pathlib import Path
from typing import Any

import pytest

from arxiv_agent.eval.dataset import EvalItem, Evidence, check_dataset, load_items

ITEM: dict[str, Any] = {
    "id": "t01",
    "question": "What does ReAct interleave?",
    "reference_answer": "Reasoning traces and task-specific actions.",
    "evidence": [
        {
            "arxiv_id": "2210.03629v3",
            "quote": "generate both reasoning traces and task-specific actions "
            "in an interleaved manner",
        }
    ],
    "kind": "single",
    "names_method": True,
}


def write(tmp_path: Path, *items: dict[str, Any]) -> Path:
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    return path


def test_loads_an_item(tmp_path: Path) -> None:
    [item] = load_items(write(tmp_path, ITEM))
    assert item == EvalItem(
        id="t01",
        question="What does ReAct interleave?",
        reference_answer="Reasoning traces and task-specific actions.",
        evidence=(Evidence("2210.03629v3", ITEM["evidence"][0]["quote"]),),
        kind="single",
        names_method=True,
    )


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_items(write(tmp_path, ITEM, ITEM))


def test_unknown_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind"):
        load_items(write(tmp_path, {**ITEM, "kind": "trivia"}))


def test_short_quotes_are_rejected(tmp_path: Path) -> None:
    item = {**ITEM, "evidence": [{"arxiv_id": "2210.03629v3", "quote": "reasoning traces"}]}
    with pytest.raises(ValueError, match="quote"):
        load_items(write(tmp_path, item))


def test_items_need_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence"):
        load_items(write(tmp_path, {**ITEM, "evidence": []}))


def test_unversioned_evidence_ids_are_rejected(tmp_path: Path) -> None:
    item = {**ITEM, "evidence": [{**ITEM["evidence"][0], "arxiv_id": "2210.03629"}]}
    with pytest.raises(ValueError, match="version"):
        load_items(write(tmp_path, item))


def test_check_dataset_accepts_a_quote_present_in_the_paper(tmp_path: Path) -> None:
    [item] = load_items(write(tmp_path, ITEM))
    text = (
        "LLMs generate both reasoning\ntraces and task-specific actions in an interleaved manner."
    )
    assert check_dataset([item], {"2210.03629v3": text}) == []


def test_check_dataset_reports_a_missing_quote_and_a_missing_paper(tmp_path: Path) -> None:
    [item] = load_items(write(tmp_path, ITEM))
    assert "not found" in check_dataset([item], {"2210.03629v3": "something else"})[0]
    assert "no text" in check_dataset([item], {})[0]
