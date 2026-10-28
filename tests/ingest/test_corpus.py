from pathlib import Path

import pytest

from arxiv_agent.documents import PaperMeta
from arxiv_agent.ingest.corpus import TitleMismatchError, check_title, load_corpus

MANIFEST = Path(__file__).resolve().parents[2] / "corpus" / "agent-papers.toml"


def meta(title: str) -> PaperMeta:
    return PaperMeta("2210.03629v3", title, (), "2022-10-06", "", "")


def test_shipped_manifest_has_at_least_fifty_versioned_papers() -> None:
    entries = load_corpus(MANIFEST)
    assert len(entries) >= 50
    assert all("v" in e.id for e in entries)
    assert len({e.id.split("v")[0] for e in entries}) == len(entries)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    entry = '[[paper]]\nid = "2210.03629v3"\ntitle = "A"\n\n'
    path.write_text(entry * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(path)


def test_unversioned_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[[paper]]\nid = "2210.03629"\ntitle = "A"\n')
    with pytest.raises(ValueError, match="version"):
        load_corpus(path)


def test_title_check_ignores_case_and_whitespace() -> None:
    check_title(meta("ReAct:  Synergizing\nReasoning"), "react: synergizing reasoning")


def test_title_check_raises_on_a_different_paper() -> None:
    with pytest.raises(TitleMismatchError, match="Toolformer"):
        check_title(meta("ReAct"), "Toolformer")
