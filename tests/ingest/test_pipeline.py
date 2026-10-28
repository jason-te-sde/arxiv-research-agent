from pathlib import Path

import httpx
import pytest

from arxiv_agent.ingest.arxiv_client import ArxivClient
from arxiv_agent.ingest.pipeline import Ingestor, local_meta
from arxiv_agent.retrieval.store import PaperStore
from tests.fakes import HashingEmbeddings, meta
from tests.pdfgen import make_pdf

FEED = (Path(__file__).parent.parent / "fixtures" / "arxiv_query.xml").read_bytes()
BODY = ["Reasoning traces help the model act in the world." for _ in range(30)]


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    return PaperStore(tmp_path / "chroma", HashingEmbeddings(), "test")


def ingestor(store: PaperStore, tmp_path: Path, client: ArxivClient | None = None) -> Ingestor:
    return Ingestor(store, words, size=40, overlap=5, client=client, pdf_dir=tmp_path / "pdfs")


def arxiv(pdf: bytes, calls: list[str] | None = None) -> ArxivClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        if "export.arxiv.org" in str(request.url):
            return httpx.Response(200, content=FEED)
        return httpx.Response(200, content=pdf)

    return ArxivClient(httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)


def test_ingest_pdf_stores_chunks(store: PaperStore, tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(make_pdf([BODY, BODY]))
    result = ingestor(store, tmp_path).ingest_pdf(pdf, meta())
    assert result.error is None
    assert result.chunks > 1
    assert store.papers()[0].chunks == result.chunks


def test_reingesting_replaces_rather_than_duplicates(store: PaperStore, tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(make_pdf([BODY]))
    ing = ingestor(store, tmp_path)
    first = ing.ingest_pdf(pdf, meta()).chunks
    ing.ingest_pdf(pdf, meta())
    assert store.papers()[0].chunks == first


def test_pdf_without_text_is_an_error_and_stores_nothing(store: PaperStore, tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(make_pdf([[]]))
    result = ingestor(store, tmp_path).ingest_pdf(pdf, meta())
    assert result.error is not None
    assert result.chunks == 0
    assert store.papers() == []


def test_ingest_arxiv_downloads_and_stores(store: PaperStore, tmp_path: Path) -> None:
    client = arxiv(make_pdf([BODY]))
    [result] = ingestor(store, tmp_path, client).ingest_arxiv(["2210.03629v3"])
    assert result.error is None
    assert result.title.startswith("ReAct")
    assert store.has_paper("2210.03629v3")


def test_title_mismatch_is_an_error(store: PaperStore, tmp_path: Path) -> None:
    client = arxiv(make_pdf([BODY]))
    [result] = ingestor(store, tmp_path, client).ingest_arxiv(
        ["2210.03629v3"], expected_titles={"2210.03629v3": "Toolformer"}
    )
    assert result.error is not None
    assert "Toolformer" in result.error
    assert not store.has_paper("2210.03629v3")


def test_skip_existing_does_not_download_again(store: PaperStore, tmp_path: Path) -> None:
    calls: list[str] = []
    ing = ingestor(store, tmp_path, arxiv(make_pdf([BODY]), calls))
    ing.ingest_arxiv(["2210.03629v3"])
    downloads = sum("arxiv.org/pdf" in c for c in calls)
    [again] = ing.ingest_arxiv(["2210.03629v3"], skip_existing=True)
    assert again.skipped
    assert sum("arxiv.org/pdf" in c for c in calls) == downloads


def test_api_failure_reports_every_id(store: PaperStore, tmp_path: Path) -> None:
    client = ArxivClient(
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))),
        min_interval=0,
    )
    results = ingestor(store, tmp_path, client).ingest_arxiv(["2210.03629v3", "2302.04761v1"])
    assert [r.arxiv_id for r in results] == ["2210.03629v3", "2302.04761v1"]
    assert all(r.error for r in results)


def test_ingest_arxiv_without_a_client_is_refused(store: PaperStore, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="arXiv client"):
        ingestor(store, tmp_path).ingest_arxiv(["2210.03629v3"])


def test_local_meta_ids_are_content_addressed() -> None:
    a = local_meta(b"%PDF-1 one", "My Paper")
    assert a.arxiv_id.startswith("local-")
    assert a.arxiv_id == local_meta(b"%PDF-1 one", "Other title").arxiv_id
    assert a.arxiv_id != local_meta(b"%PDF-1 two", "My Paper").arxiv_id
    assert local_meta(b"x", "  Two\nlines ").title == "Two lines"


def test_title_too_long_for_the_budget_is_an_error(store: PaperStore, tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(make_pdf([BODY]))
    result = ingestor(store, tmp_path).ingest_pdf(pdf, meta(title="word " * 50))
    assert result.error is not None
    assert "title" in result.error
