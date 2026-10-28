"""Citations are chunk ids in square brackets; only ids the tools returned count."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from arxiv_agent.documents import Chunk

_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
# An arXiv id ("2210.03629v3") or an upload id ("local-0123456789ab"), then "#index".
_CHUNK_ID = re.compile(r"(?:\d{4}\.\d{4,5}v\d+|local-[0-9a-f]{12})#\d+")


@dataclass(frozen=True)
class Citation:
    id: str
    arxiv_id: str
    title: str
    page: int


def extract_citations(text: str) -> list[str]:
    found: list[str] = []
    for group in _BRACKETS.findall(text):
        for piece in re.split(r"[;,]", group):
            candidate = piece.strip()
            if _CHUNK_ID.fullmatch(candidate) and candidate not in found:
                found.append(candidate)
    return found


def verify_citations(
    cited: Sequence[str], seen: Mapping[str, Chunk]
) -> tuple[list[Citation], list[str]]:
    """Split cited ids into those a tool returned this run and those it did not."""
    verified = [
        Citation(c, seen[c].arxiv_id, seen[c].title, seen[c].page) for c in cited if c in seen
    ]
    return verified, [c for c in cited if c not in seen]
