import sys
from collections.abc import Sequence
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult

from arxiv_agent.mcp_server import MAX_CHUNK_ID_CHARS, MAX_K, MAX_QUERY_CHARS, create_server
from arxiv_agent.retrieval.retriever import TwoStageRetriever
from arxiv_agent.retrieval.store import PaperStore
from mcp import Client, StdioServerParameters
from tests.fakes import HashingEmbeddings, chunk


class LengthReranker:
    """Prefers longer passages, so the rerank order differs from the dense order."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(len(t)) for t in texts]


def seeded_store() -> PaperStore:
    store = PaperStore(None, HashingEmbeddings(), "mcp-test")
    store.replace_paper(
        [
            chunk("1111.11111v2", 0, "alpha beta gamma", title="First", page=1),
            chunk("1111.11111v2", 1, "alpha beta gamma delta epsilon zeta", title="First", page=3),
        ]
    )
    store.replace_paper([chunk("2222.22222v1", 0, "alpha eta theta", title="Second", page=2)])
    store.replace_paper([chunk("local-0123456789ab", 0, "iota kappa", title="Upload", page=1)])
    return store


def build() -> MCPServer:
    store = seeded_store()
    retriever = TwoStageRetriever(store, LengthReranker(), k_candidates=MAX_K, k=MAX_K)
    return create_server(store, retriever)


async def call(tool: str, arguments: dict[str, Any]) -> CallToolResult:
    # One in-memory client per call: an async fixture would enter and leave the
    # client's cancel scope in different tasks.
    async with Client(build()) as c:
        return await c.call_tool(tool, arguments)


async def test_exactly_the_three_read_only_tools() -> None:
    async with Client(build()) as client:
        tools = (await client.list_tools()).tools
    assert sorted(t.name for t in tools) == ["get_passage", "list_papers", "search_papers"]
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False


async def test_search_returns_reranked_passages_with_identity() -> None:
    result = await call("search_papers", {"query": "alpha beta", "k": 2})
    assert not result.is_error
    passages = result.structured_content["result"]
    assert [p["chunk_id"] for p in passages] == ["1111.11111v2#1", "1111.11111v2#0"]
    top = passages[0]
    assert top["paper_id"] == "1111.11111"
    assert top["version"] == "v2"
    assert top["title"] == "First"
    assert top["page"] == 3
    assert top["text"] == "alpha beta gamma delta epsilon zeta"
    assert top["score"] > passages[1]["score"]


async def test_k_limits_the_number_of_passages() -> None:
    result = await call("search_papers", {"query": "alpha", "k": 1})
    assert len(result.structured_content["result"]) == 1
    default = await call("search_papers", {"query": "alpha"})
    assert len(default.structured_content["result"]) == 4


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * (MAX_QUERY_CHARS + 1)},
        {"query": "alpha", "k": 0},
        {"query": "alpha", "k": MAX_K + 1},
        {"query": "alpha", "k": "many"},
        {},
    ],
)
async def test_search_rejects_inputs_out_of_bounds(arguments: dict[str, Any]) -> None:
    result = await call("search_papers", arguments)
    assert result.is_error


async def test_search_accepts_the_bounds_themselves() -> None:
    result = await call("search_papers", {"query": "x" * MAX_QUERY_CHARS, "k": MAX_K})
    assert not result.is_error


async def test_get_passage_returns_text_and_citation() -> None:
    result = await call("get_passage", {"chunk_id": "1111.11111v2#1"})
    assert not result.is_error
    got = result.structured_content
    assert got is not None
    assert got["text"] == "alpha beta gamma delta epsilon zeta"
    assert got["citation"] == "[1111.11111v2#1] First, p.3"
    assert (got["paper_id"], got["version"]) == ("1111.11111", "v2")


@pytest.mark.parametrize(
    "chunk_id", ["9999.99999v1#0", "1111.11111v2#99", "not an id", "1111.11111v2", "../x#1"]
)
async def test_get_passage_rejects_unknown_or_malformed_ids(chunk_id: str) -> None:
    assert (await call("get_passage", {"chunk_id": chunk_id})).is_error


async def test_get_passage_bounds_the_id_length() -> None:
    long_id = "a" * MAX_CHUNK_ID_CHARS + "#1"
    assert (await call("get_passage", {"chunk_id": long_id})).is_error


async def test_list_papers_lists_every_paper() -> None:
    result = await call("list_papers", {})
    assert not result.is_error
    papers = result.structured_content["result"]
    assert papers == [
        {"paper_id": "1111.11111", "version": "v2", "title": "First", "chunks": 2},
        {"paper_id": "2222.22222", "version": "v1", "title": "Second", "chunks": 1},
        {"paper_id": "local-0123456789ab", "version": None, "title": "Upload", "chunks": 1},
    ]


async def test_tools_leave_the_index_unchanged() -> None:
    store = seeded_store()
    retriever = TwoStageRetriever(store, LengthReranker(), k_candidates=MAX_K, k=MAX_K)
    before = (store.papers(), store.all_embeddings()[0])
    async with Client(create_server(store, retriever)) as c:
        await c.call_tool("search_papers", {"query": "alpha"})
        await c.call_tool("get_passage", {"chunk_id": "2222.22222v1#0"})
        await c.call_tool("list_papers", {})
    assert (store.papers(), store.all_embeddings()[0]) == before


STDIO_SERVER = """
from arxiv_agent.mcp_server import create_server
from arxiv_agent.retrieval.retriever import DenseRetriever
from tests.mcp.test_mcp_server import seeded_store
store = seeded_store()
create_server(store, DenseRetriever(store, k=20)).run("stdio")
"""


@pytest.mark.parametrize(("mode", "version"), [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
async def test_stdio_serves_both_spec_revisions(mode: str, version: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-c", STDIO_SERVER])
    async with Client(params, mode=mode) as c:
        assert c.protocol_version == version
        result = await c.call_tool("search_papers", {"query": "iota kappa", "k": 1})
        assert result.structured_content["result"][0]["chunk_id"] == "local-0123456789ab#0"
