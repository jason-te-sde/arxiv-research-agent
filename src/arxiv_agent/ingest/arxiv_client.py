"""A small client for the arXiv export API and PDF downloads.

arXiv asks API users to wait three seconds between requests and to keep one
connection at a time. Every request this client makes, metadata and PDFs
alike, holds one lock across the wait and the request itself, so concurrent
callers (the API's threadpool, the agent's fetch tool) queue behind it.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from xml.etree.ElementTree import Element

import httpx
from defusedxml import ElementTree

from arxiv_agent import __version__
from arxiv_agent.documents import PaperMeta

API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = (
    f"arxiv-research-agent/{__version__} (+https://github.com/jason-te-sde/arxiv-research-agent)"
)
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_BATCH = 50


class ArxivError(Exception):
    """The arXiv API or the PDF download failed."""


class PaperNotFoundError(ArxivError):
    """arXiv returned no entry for a requested identifier."""


def normalize_arxiv_id(raw: str) -> str:
    """Accept a bare id, ``arXiv:`` prefix, or abs/pdf URL; return ``YYMM.NNNNN[vN]``."""
    text = raw.strip()
    text = re.sub(
        r"^(?:arxiv:|https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/)", "", text, flags=re.I
    )
    text = re.sub(r"\.pdf$", "", text)
    if not _ID.fullmatch(text):
        raise ValueError(f"not a new-style arXiv identifier: {raw!r}")
    return text


def base_id(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def make_http_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True)


class ArxivClient:
    def __init__(
        self,
        http: httpx.Client,
        *,
        min_interval: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        api_url: str = API_URL,
    ) -> None:
        self._http = http
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._api_url = api_url
        self._last: float | None = None
        self._lock = threading.Lock()

    def get(self, ids: Sequence[str]) -> list[PaperMeta]:
        wanted = [normalize_arxiv_id(i) for i in ids]
        found: dict[str, PaperMeta] = {}
        for start in range(0, len(wanted), _BATCH):
            batch = wanted[start : start + _BATCH]
            for meta in self._query({"id_list": ",".join(batch), "max_results": str(len(batch))}):
                found[meta.arxiv_id] = meta
                found.setdefault(base_id(meta.arxiv_id), meta)
        missing = [i for i in wanted if i not in found]
        if missing:
            raise PaperNotFoundError(f"arXiv returned no entry for: {', '.join(missing)}")
        return [found[i] for i in wanted]

    def search(self, query: str, max_results: int = 5) -> list[PaperMeta]:
        # Only word characters reach the query: quotes and parentheses are arXiv syntax.
        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query)
        if not terms:
            raise ValueError("search query has no words")
        return self._query(
            {
                "search_query": " AND ".join(f"all:{t}" for t in terms),
                "max_results": str(max_results),
                "sortBy": "relevance",
            }
        )

    def download_pdf(self, meta: PaperMeta, dest: Path) -> Path:
        path = dest / f"{meta.arxiv_id}.pdf"
        if path.exists() and path.read_bytes()[:4] == b"%PDF":
            return path
        response = self._request(meta.pdf_url, params=None)
        if response.content[:4] != b"%PDF":
            raise ArxivError(f"{meta.pdf_url} returned something that is not a PDF")
        dest.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".pdf.part")
        partial.write_bytes(response.content)
        partial.replace(path)
        return path

    def _query(self, params: dict[str, str]) -> list[PaperMeta]:
        response = self._request(self._api_url, params=params)
        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            raise ArxivError(f"unparseable arXiv response: {exc}") from exc
        return [_parse_entry(e) for e in root.findall("a:entry", _ATOM) if _is_paper(e)]

    def _request(self, url: str, params: dict[str, str] | None) -> httpx.Response:
        with self._lock:
            if self._last is not None:
                wait = self._min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            try:
                response = self._http.get(url, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ArxivError(f"request to {url} failed: {exc}") from exc
            finally:
                self._last = self._clock()
            return response


def _text(entry: Element, tag: str) -> str:
    node = entry.find(f"a:{tag}", _ATOM)
    return " ".join((node.text or "").split()) if node is not None else ""


def _is_paper(entry: Element) -> bool:
    # Invalid ids come back as an entry whose id is an API error URL.
    return _ID.search(_text(entry, "id").rsplit("/", 1)[-1]) is not None


def _parse_entry(entry: Element) -> PaperMeta:
    arxiv_id = _text(entry, "id").rsplit("/", 1)[-1]
    pdf_url = next(
        (
            link.get("href", "")
            for link in entry.findall("a:link", _ATOM)
            if link.get("title") == "pdf"
        ),
        f"https://arxiv.org/pdf/{arxiv_id}",
    )
    return PaperMeta(
        arxiv_id=arxiv_id,
        title=_text(entry, "title"),
        authors=tuple(
            " ".join((n.text or "").split()) for n in entry.findall("a:author/a:name", _ATOM)
        ),
        published=_text(entry, "published")[:10],
        abstract=_text(entry, "summary"),
        pdf_url=pdf_url,
    )
