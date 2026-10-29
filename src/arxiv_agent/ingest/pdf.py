"""PDF text extraction and the cleaning that makes it chunkable.

Text comes out of ``pypdf`` one layout line at a time. Cleaning turns those
lines back into paragraphs (so the splitter can prefer paragraph boundaries),
undoes line-break hyphenation and ligatures, drops the arXiv side stamp, bare
page numbers and lines of figure-font garbage, and cuts the reference list,
which otherwise floods retrieval with author names and titles that match
almost any query.
"""

from __future__ import annotations

import bisect
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class PdfExtractionError(Exception):
    """The file could not be read as a PDF."""


class EmptyPdfError(PdfExtractionError):
    """The PDF has no extractable text (for example, a scanned paper)."""


@dataclass(frozen=True)
class PaperText:
    text: str
    page_starts: tuple[int, ...]

    def page_of(self, offset: int) -> int:
        """1-based page number holding character ``offset`` of ``text``."""
        return max(1, bisect.bisect_right(self.page_starts, offset))


_ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}(v\d+)?\s+\[[^\]]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}$")
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_REFERENCES_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)[ \t]*$",
    re.MULTILINE,
)
_SENTENCE_END = (".", "?", "!", ":")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Figures typeset with subset fonts often extract as shifted letters separated
# by control bytes; a line with this many of them is not prose.
_GARBAGE_CONTROLS = 3
# A line shorter than this fraction of the page's median line ends its paragraph
# when it also ends a sentence: it is the ragged last line of that paragraph.
_SHORT_LINE = 0.75
# A references heading starts the bibliography only if citations follow it; a
# year within this many characters tells it apart from a table-of-contents line.
# (Position in the text does not: long appendices put the bibliography at 15%.)
_BIBLIOGRAPHY_WINDOW = 2000
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")


def clean_page(raw: str) -> str:
    # NFKC folds ligatures (U+FB01 'fi') that PDF fonts emit as one glyph.
    lines = [
        _CONTROL.sub("", ln).strip()
        # split on "\n" only: splitlines() also breaks on form feeds and other controls
        for ln in unicodedata.normalize("NFKC", raw).split("\n")
        if len(_CONTROL.findall(ln)) < _GARBAGE_CONTROLS
    ]
    lines = [ln for ln in lines if not _ARXIV_STAMP.match(ln) and not _PAGE_NUMBER.match(ln)]
    lengths = [len(ln) for ln in lines if ln]
    if not lengths:
        return ""
    short = _SHORT_LINE * statistics.median(lengths)

    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if not line:
            if current:
                paragraphs.append(current)
            current = ""
            continue
        if not current:
            current = line
        elif current.endswith("-") and len(current) > 1 and current[-2].isalpha():
            # "reason-" + "ing" is one word; "GPT-" + "4" keeps its hyphen.
            current = current[:-1] + line if line[0].islower() else current + line
        else:
            current = f"{current} {line}"
        if len(line) < short and line.endswith(_SENTENCE_END):
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return "\n\n".join(" ".join(p.split()) for p in paragraphs)


def strip_references(text: str) -> str:
    """Cut ``text`` at its last references heading that a bibliography follows."""
    matches = [
        m
        for m in _REFERENCES_HEADING.finditer(text)
        if _YEAR.search(text, m.end(), m.end() + _BIBLIOGRAPHY_WINDOW)
    ]
    if not matches:
        return text
    return text[: matches[-1].start()].rstrip()


def extract_text(path: Path) -> PaperText:
    try:
        reader = PdfReader(path)
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises many types on malformed input
        raise PdfExtractionError(f"{path}: {exc}") from exc

    # Find the bibliography on the raw text, where headings are still lines of
    # their own, then keep only the pages (and part of a page) before it.
    separator = "\n"
    kept = len(strip_references(separator.join(raw_pages)))
    pages: list[str] = []
    position = 0
    for raw in raw_pages:
        if position >= kept:
            break
        pages.append(raw[: kept - position])
        position += len(raw) + len(separator)

    text = ""
    starts: list[int] = []
    for page in pages:
        cleaned = clean_page(page)
        if cleaned and text:
            text += "\n\n"
        starts.append(len(text))
        text += cleaned
    if not text.strip():
        raise EmptyPdfError(f"{path}: no extractable text")
    return PaperText(text=text, page_starts=tuple(starts))
