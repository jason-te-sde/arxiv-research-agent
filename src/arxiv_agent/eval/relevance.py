"""Model-free relevance: a chunk is relevant if it holds most of an evidence quote."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher

from arxiv_agent.documents import Chunk
from arxiv_agent.eval.dataset import Evidence
from arxiv_agent.retrieval.hits import Hit

# Share of the quote that must appear contiguously in the chunk. Below 1.0 so a
# quote cut by a chunk boundary still counts in the chunk that holds most of it.
# Chunk overlap does not guarantee such a chunk exists (the splitter carries
# over only whole pieces shorter than the overlap); `eval check-dataset` does,
# by refusing a question set in which some quote has no such chunk.
THRESHOLD = 0.6

# Curly quotes and en/em dashes fold to their ASCII forms.
_FOLD = str.maketrans(
    {0x201C: '"', 0x201D: '"', 0x2018: "'", 0x2019: "'", 0x2014: "-", 0x2013: "-"}
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_FOLD)
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, evidence: Evidence, threshold: float = THRESHOLD) -> bool:
    if chunk.arxiv_id != evidence.arxiv_id:
        return False
    quote = normalize(evidence.quote)
    text = normalize(chunk.text)
    if quote in text:
        return True
    match = SequenceMatcher(None, text, quote, autojunk=False).find_longest_match(
        0, len(text), 0, len(quote)
    )
    return match.size >= threshold * len(quote)


def first_relevant_rank(hits: Sequence[Hit], evidence: Sequence[Evidence], k: int) -> int | None:
    for rank, hit in enumerate(hits[:k], start=1):
        if any(is_relevant(hit.chunk, ev) for ev in evidence):
            return rank
    return None


def reciprocal_rank(hits: Sequence[Hit], evidence: Sequence[Evidence], k: int) -> float:
    rank = first_relevant_rank(hits, evidence, k)
    return 0.0 if rank is None else 1.0 / rank


def recall_at_k(hits: Sequence[Hit], evidence: Sequence[Evidence], k: int) -> float:
    found = sum(1 for ev in evidence if any(is_relevant(h.chunk, ev) for h in hits[:k]))
    return found / len(evidence)
