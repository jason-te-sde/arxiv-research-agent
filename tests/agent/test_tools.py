from pathlib import Path

import httpx
import pytest
from langchain_core.tools import BaseTool

from arxiv_agent.agent.tools import ToolContext, build_tools, format_hits
from arxiv_agent.ingest.arxiv_client import ArxivClient
from arxiv_agent.ingest.pipeline import Ingestor
from arxiv_agent.retrieval.hits import Hit
from arxiv_agent.retrieval.retriever import DenseRetriever
from arxiv_agent.retrieval.store import PaperStore
from tests.agent.scripted import ScriptedChatModel, say
from tests.fakes import HashingEmbeddings, chunk
from tests.pdfgen import make_pdf

REACT = "2210.03629v3"
REFLEXION = "2303.11366v4"
FEED = (Path(__file__).parent.parent / "fixtures" / "arxiv_query.xml").read_bytes()


@pytest.fixture
def store(tmp_path: Path) -> PaperStore:
    s = PaperStore(tmp_path / "chroma", HashingEmbeddings(), "tools")
    s.replace_paper(
        [
            chunk(REACT, 0, "reasoning traces and actions interleaved", title="ReAct"),
            chunk(REACT, 1, "hotpotqa fever alfworld webshop results", title="ReAct"),
        ]
    )
    s.replace_paper(
        [
            chunk(REFLEXION, 0, "verbal reinforcement reflection memory", title="Reflexion"),
            chunk(REFLEXION, 1, "reflection improves humaneval pass rate", title="Reflexion"),
        ]
    )
    return s


def context(
    store: PaperStore,
    model: ScriptedChatModel | None = None,
    ingestor: Ingestor | None = None,
    client: ArxivClient | None = None,
) -> ToolContext:
    return ToolContext(
        retriever=DenseRetriever(store, k=2),
        store=store,
        model=model or ScriptedChatModel(script=[]),
        ingestor=ingestor,
        client=client,
    )


def tool(ctx: ToolContext, name: str) -> BaseTool:
    return {t.name: t for t in build_tools(ctx)}[name]


def test_the_four_tools_have_their_names(store: PaperStore) -> None:
    names = [t.name for t in build_tools(context(store))]
    assert names == ["search_papers", "summarize_paper", "fetch_paper", "compare_papers"]


def test_format_hits_heads_each_passage_with_its_id() -> None:
    text = format_hits([Hit(chunk(REACT, 4, "body text", title="ReAct", page=3), 0.9)])
    assert text == f"[{REACT}#4] ReAct, p.3\nbody text"


async def test_search_records_what_it_returned(store: PaperStore) -> None:
    ctx = context(store)
    out = await tool(ctx, "search_papers").ainvoke({"query": "reflection memory"})
    assert f"[{REFLEXION}#0]" in out
    assert f"{REFLEXION}#0" in ctx.seen
    assert ctx.contexts[0] == "Reflexion\nverbal reinforcement reflection memory"


async def test_repeated_passages_are_recorded_once(store: PaperStore) -> None:
    ctx = context(store)
    search = tool(ctx, "search_papers")
    await search.ainvoke({"query": "reflection memory"})
    await search.ainvoke({"query": "reflection memory"})
    assert len(ctx.contexts) == len(set(ctx.contexts)) == 2


async def test_search_inside_one_paper(store: PaperStore) -> None:
    ctx = context(store)
    await tool(ctx, "search_papers").ainvoke({"query": "results", "arxiv_id": REACT})
    assert {c.arxiv_id for c in ctx.seen.values()} == {REACT}


async def test_search_accepts_an_unversioned_id(store: PaperStore) -> None:
    ctx = context(store)
    out = await tool(ctx, "search_papers").ainvoke({"query": "results", "arxiv_id": "2210.03629"})
    assert f"[{REACT}#" in out


async def test_search_in_a_paper_that_is_not_ingested_says_so(store: PaperStore) -> None:
    out = await tool(context(store), "search_papers").ainvoke(
        {"query": "x", "arxiv_id": "2401.15884v3"}
    )
    assert "not ingested" in out
    assert "fetch_paper" in out


async def test_summarize_reads_only_that_paper(store: PaperStore) -> None:
    model = ScriptedChatModel(script=[say("Problem: ... [2210.03629v3#0]")])
    ctx = context(store, model)
    out = await tool(ctx, "summarize_paper").ainvoke({"arxiv_id": REACT})
    assert out.startswith("Problem:")
    prompt = model.inputs[0][-1].text
    assert f"{REACT}#0" in prompt
    assert REFLEXION not in prompt
    assert {c.arxiv_id for c in ctx.seen.values()} == {REACT}


async def test_summarize_unknown_paper(store: PaperStore) -> None:
    out = await tool(context(store), "summarize_paper").ainvoke({"arxiv_id": "2401.15884v3"})
    assert "not ingested" in out


async def test_compare_sends_passages_from_every_paper(store: PaperStore) -> None:
    model = ScriptedChatModel(script=[say("ReAct ... Reflexion ...")])
    ctx = context(store, model)
    await tool(ctx, "compare_papers").ainvoke(
        {"arxiv_ids": [REACT, REFLEXION], "aspect": "how they use feedback"}
    )
    prompt = model.inputs[0][-1].text
    assert f"[{REACT}#" in prompt
    assert f"[{REFLEXION}#" in prompt
    assert "how they use feedback" in prompt


async def test_compare_needs_two_papers(store: PaperStore) -> None:
    out = await tool(context(store), "compare_papers").ainvoke(
        {"arxiv_ids": [REACT], "aspect": "x"}
    )
    assert "at least two" in out


async def test_compare_names_the_papers_that_are_missing(store: PaperStore) -> None:
    out = await tool(context(store), "compare_papers").ainvoke(
        {"arxiv_ids": [REACT, "2401.15884v3"], "aspect": "x"}
    )
    assert "2401.15884v3" in out
    assert "not ingested" in out


def fetching(store: PaperStore, tmp_path: Path, calls: list[str]) -> ToolContext:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "export.arxiv.org" in str(request.url):
            return httpx.Response(200, content=FEED)
        return httpx.Response(200, content=make_pdf([["Toolformer teaches itself to use tools."]]))

    client = ArxivClient(httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)
    ingestor = Ingestor(
        store, lambda t: len(t.split()), 40, 5, client=client, pdf_dir=tmp_path / "pdfs"
    )
    return context(store, ingestor=ingestor, client=client)


async def test_fetch_needs_an_id_or_a_query(store: PaperStore, tmp_path: Path) -> None:
    out = await tool(fetching(store, tmp_path, []), "fetch_paper").ainvoke({})
    assert "arxiv_id or a query" in out


async def test_fetch_of_a_stored_paper_downloads_nothing(store: PaperStore, tmp_path: Path) -> None:
    calls: list[str] = []
    out = await tool(fetching(store, tmp_path, calls), "fetch_paper").ainvoke({"arxiv_id": REACT})
    assert "already ingested" in out
    assert calls == []


async def test_fetch_ingests_a_new_paper(store: PaperStore, tmp_path: Path) -> None:
    out = await tool(fetching(store, tmp_path, []), "fetch_paper").ainvoke(
        {"arxiv_id": "2302.04761v1"}
    )
    assert "Ingested" in out
    assert store.has_paper("2302.04761v1")


async def test_fetch_by_query_ingests_the_top_result(store: PaperStore, tmp_path: Path) -> None:
    out = await tool(fetching(store, tmp_path, []), "fetch_paper").ainvoke(
        {"query": "language models tools"}
    )
    assert "Ingested" in out
    assert "2302.04761v1" in out


async def test_fetch_without_network_access_is_disabled(store: PaperStore) -> None:
    out = await tool(context(store), "fetch_paper").ainvoke({"arxiv_id": "2302.04761v1"})
    assert "disabled" in out


def test_descriptions_carry_no_source_indentation(store: PaperStore) -> None:
    for t in build_tools(context(store)):
        assert "\n " not in t.description, t.name
