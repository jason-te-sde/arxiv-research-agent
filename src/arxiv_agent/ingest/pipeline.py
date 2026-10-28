"""From a PDF (local or on arXiv) to stored, embedded chunks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from arxiv_agent.documents import PaperMeta
from arxiv_agent.ingest.arxiv_client import ArxivClient, ArxivError
from arxiv_agent.ingest.chunking import LengthFunction, chunk_paper
from arxiv_agent.ingest.corpus import TitleMismatchError, check_title
from arxiv_agent.ingest.pdf import PdfExtractionError, extract_text
from arxiv_agent.retrieval.store import PaperStore


@dataclass(frozen=True)
class IngestResult:
    arxiv_id: str
    title: str
    chunks: int
    error: str | None = None
    skipped: bool = False


def local_meta(pdf: bytes, title: str) -> PaperMeta:
    """Metadata for an uploaded PDF: the id is derived from the file's content."""
    digest = hashlib.sha256(pdf).hexdigest()[:12]
    return PaperMeta(
        arxiv_id=f"local-{digest}",
        title=" ".join(title.split()),
        authors=(),
        published="",
        abstract="",
        pdf_url="",
    )


class Ingestor:
    def __init__(
        self,
        store: PaperStore,
        length: LengthFunction,
        size: int,
        overlap: int,
        client: ArxivClient | None,
        pdf_dir: Path,
    ) -> None:
        self._store = store
        self._length = length
        self._size = size
        self._overlap = overlap
        self._client = client
        self._pdf_dir = pdf_dir

    def ingest_pdf(self, path: Path, meta: PaperMeta) -> IngestResult:
        try:
            paper = extract_text(path)
        except PdfExtractionError as exc:
            return IngestResult(meta.arxiv_id, meta.title, 0, error=str(exc))
        try:
            chunks = chunk_paper(paper, meta, self._length, self._size, self._overlap)
        except ValueError as exc:  # the title leaves no room in the token budget
            return IngestResult(meta.arxiv_id, meta.title, 0, error=str(exc))
        stored = self._store.replace_paper(chunks)
        return IngestResult(meta.arxiv_id, meta.title, stored)

    def ingest_arxiv(
        self,
        ids: Sequence[str],
        expected_titles: Mapping[str, str] | None = None,
        *,
        skip_existing: bool = False,
    ) -> list[IngestResult]:
        if self._client is None:
            raise RuntimeError("ingesting from arXiv needs an arXiv client")
        try:
            metas = self._client.get(ids)
        except ArxivError as exc:
            return [IngestResult(i, "", 0, error=str(exc)) for i in ids]
        return [self._ingest_one(self._client, m, expected_titles, skip_existing) for m in metas]

    def _ingest_one(
        self,
        client: ArxivClient,
        meta: PaperMeta,
        expected_titles: Mapping[str, str] | None,
        skip_existing: bool,
    ) -> IngestResult:
        try:
            if expected_titles is not None and meta.arxiv_id in expected_titles:
                check_title(meta, expected_titles[meta.arxiv_id])
        except TitleMismatchError as exc:
            return IngestResult(meta.arxiv_id, meta.title, 0, error=str(exc))
        if skip_existing and self._store.has_paper(meta.arxiv_id):
            return IngestResult(meta.arxiv_id, meta.title, 0, skipped=True)
        try:
            path = client.download_pdf(meta, self._pdf_dir)
        except ArxivError as exc:
            return IngestResult(meta.arxiv_id, meta.title, 0, error=str(exc))
        return self.ingest_pdf(path, meta)
