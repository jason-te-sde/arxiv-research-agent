from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from arxiv_agent.ingest.arxiv_client import (
    ArxivClient,
    ArxivError,
    PaperNotFoundError,
    normalize_arxiv_id,
)

FIXTURE = (Path(__file__).parent.parent / "fixtures" / "arxiv_query.xml").read_bytes()


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], clock: FakeClock | None = None
) -> ArxivClient:
    clock = clock or FakeClock()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return ArxivClient(http, clock=clock, sleep=clock.sleep)


def feed(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=FIXTURE)


def test_get_parses_the_atom_entries() -> None:
    papers = client_for(feed).get(["2210.03629v3", "2302.04761v1"])
    react = papers[0]
    assert react.arxiv_id == "2210.03629v3"
    assert react.title == "ReAct: Synergizing Reasoning and Acting in Language Models"
    assert react.authors[0] == "Shunyu Yao"
    assert react.published == "2022-10-06"
    assert react.pdf_url == "https://arxiv.org/pdf/2210.03629v3"
    assert "\n" not in react.abstract
    assert react.abstract.startswith("While large language models")


def test_get_returns_papers_in_request_order() -> None:
    papers = client_for(feed).get(["2302.04761v1", "2210.03629v3"])
    assert [p.arxiv_id for p in papers] == ["2302.04761v1", "2210.03629v3"]


def test_get_sends_the_ids_to_the_api() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return feed(request)

    client_for(handler).get(["2210.03629v3", "2302.04761v1"])
    assert seen[0].url.params["id_list"] == "2210.03629v3,2302.04761v1"


def test_get_raises_for_an_id_the_api_did_not_return() -> None:
    with pytest.raises(PaperNotFoundError, match=r"2303\.11366v4"):
        client_for(feed).get(["2210.03629v3", "2303.11366v4"])


def test_unversioned_request_matches_the_returned_version() -> None:
    papers = client_for(feed).get(["2210.03629"])
    assert papers[0].arxiv_id == "2210.03629v3"


def test_requests_are_spaced_by_the_minimum_interval() -> None:
    clock = FakeClock()
    client = client_for(feed, clock)
    client.get(["2210.03629v3"])
    clock.now += 1.0
    client.get(["2210.03629v3"])
    assert clock.slept == [pytest.approx(2.0)]


def test_search_builds_an_all_fields_query() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return feed(request)

    results = client_for(handler).search("tool use agents", max_results=2)
    assert seen[0].url.params["search_query"] == "all:tool AND all:use AND all:agents"
    assert seen[0].url.params["max_results"] == "2"
    assert len(results) == 2


def test_http_error_becomes_arxiv_error() -> None:
    with pytest.raises(ArxivError):
        client_for(lambda _: httpx.Response(503)).get(["2210.03629v3"])


def test_download_writes_the_pdf(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "export.arxiv.org" in str(request.url):
            return feed(request)
        return httpx.Response(200, content=b"%PDF-1.4 body")

    client = client_for(handler)
    meta = client.get(["2210.03629v3"])[0]
    path = client.download_pdf(meta, tmp_path)
    assert path == tmp_path / "2210.03629v3.pdf"
    assert path.read_bytes().startswith(b"%PDF")


def test_download_skips_a_cached_pdf(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return feed(request)

    client = client_for(handler)
    meta = client.get(["2210.03629v3"])[0]
    (tmp_path / "2210.03629v3.pdf").write_bytes(b"%PDF-cached")
    client.download_pdf(meta, tmp_path)
    assert len(calls) == 1


def test_download_rejects_a_non_pdf_body(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "export.arxiv.org" in str(request.url):
            return feed(request)
        return httpx.Response(200, content=b"<html>captcha</html>")

    client = client_for(handler)
    meta = client.get(["2210.03629v3"])[0]
    with pytest.raises(ArxivError, match="not a PDF"):
        client.download_pdf(meta, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2210.03629v3", "2210.03629v3"),
        ("2210.03629", "2210.03629"),
        ("arXiv:2210.03629v3", "2210.03629v3"),
        ("https://arxiv.org/abs/2210.03629v3", "2210.03629v3"),
        ("https://arxiv.org/pdf/2210.03629v3.pdf", "2210.03629v3"),
        (" 2401.15884 ", "2401.15884"),
    ],
)
def test_normalize_arxiv_id(raw: str, expected: str) -> None:
    assert normalize_arxiv_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "react", "cs/0101001", "2210.036"])
def test_normalize_rejects_what_is_not_a_new_style_id(raw: str) -> None:
    with pytest.raises(ValueError, match="arXiv"):
        normalize_arxiv_id(raw)
