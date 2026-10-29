"""The pinned corpus manifest (``corpus/*.toml``) and the title check it enables."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from arxiv_agent.documents import PaperMeta
from arxiv_agent.ingest.arxiv_client import base_id, normalize_arxiv_id


class TitleMismatchError(Exception):
    """arXiv returned a different paper than the manifest names."""


@dataclass(frozen=True)
class CorpusEntry:
    id: str
    title: str


def load_corpus(path: Path) -> list[CorpusEntry]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries: list[CorpusEntry] = []
    seen: set[str] = set()
    for row in data.get("paper", []):
        arxiv_id = normalize_arxiv_id(str(row["id"]))
        if arxiv_id == base_id(arxiv_id):
            raise ValueError(f"{path}: {arxiv_id} must pin a version (e.g. {arxiv_id}v1)")
        if base_id(arxiv_id) in seen:
            raise ValueError(f"{path}: duplicate paper {base_id(arxiv_id)}")
        seen.add(base_id(arxiv_id))
        entries.append(CorpusEntry(id=arxiv_id, title=str(row["title"])))
    return entries


def _fold(title: str) -> str:
    return " ".join(title.split()).casefold()


def check_title(meta: PaperMeta, expected: str) -> None:
    if _fold(meta.title) != _fold(expected):
        raise TitleMismatchError(
            f"{meta.arxiv_id} is {meta.title!r} on arXiv, but the manifest expects {expected!r}"
        )
